"""Throughput-only CPU batching of the frozen coding study; no prompt/scorer changes."""

import argparse
import fcntl
import hashlib
import json
import resource
import time
from pathlib import Path

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from mindscape.coding.batching import evaluate_batch
from mindscape.coding.model import LocalCoder
from mindscape.v3.coding_learning.agent import CONDITIONS, solve
from mindscape.v3.coding_learning.data import load_public, load_scorer
from mindscape.v3.coding_learning.execution import Executor
from mindscape.v3.coding_learning.experiment import hashes
from mindscape.v3.coding_learning.memory import VerifiedMemory
from mindscape.v3.coding_learning.planning import ProposalEvaluator
from mindscape.v3.coding_learning.train import source_hash
from mindscape.v3.schema import digest

SYSTEM = "You are a Python debugging researcher. Return exactly the requested JSON. Use only visible evidence."


class CpuBatchCoder(LocalCoder):
    def __init__(self, path, cache):
        super().__init__(path, cache_enabled=False, device="cpu", precision="float32")
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.model_hash = digest(hashes(path))
        self.batch_records = []

    def generate_batch(self, requests):
        output = [None] * len(requests)
        pending = {}
        for i, request in enumerate(requests):
            key = digest([self.model_hash, list(request), "cpu-float32-batched-v1"])
            path = self.cache / (key + ".json")
            if path.exists():
                saved = json.loads(path.read_text())
                if saved["response_hash"] != digest(saved["response"]):
                    raise ValueError("Corrupt request cache")
                output[i] = (
                    saved["response"],
                    {
                        "cache_hit": True,
                        "tokens_in": 0,
                        "tokens_out": 0,
                        "generation": 0,
                        "original": saved["timing"],
                        "request_hash": key,
                    },
                )
            else:
                pending.setdefault(key, {"request": request, "indices": []})["indices"].append(i)
        if pending:
            keys = list(pending)
            fresh = [pending[key]["request"] for key in keys]
            limits = {r[2] for r in fresh}
            if len(limits) != 1:
                raise ValueError("Shared output-token cap required")
            texts = [
                self.tokenizer.apply_chat_template(
                    [{"role": "system", "content": r[0]}, {"role": "user", "content": r[1]}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for r in fresh
            ]
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True)
            if inputs["input_ids"].shape[1] + next(iter(limits)) > 32768:
                raise ValueError("Context exceeded; truncation forbidden")
            first = [None]
            started = time.perf_counter()

            class First(StoppingCriteria):
                def __call__(self, input_ids, scores, **kwargs):
                    if first[0] is None:
                        first[0] = time.perf_counter() - started
                    return False

            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=next(iter(limits)),
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    stopping_criteria=StoppingCriteriaList([First()]),
                )
            duration = time.perf_counter() - started
            tokens = generated[:, inputs["input_ids"].shape[1] :].tolist()
            lengths = inputs["attention_mask"].sum(dim=1).tolist()
            for key, ids, length in zip(keys, tokens, lengths, strict=True):
                if self.tokenizer.eos_token_id in ids:
                    ids = ids[: ids.index(self.tokenizer.eos_token_id) + 1]
                response = self.tokenizer.decode(ids, skip_special_tokens=True)
                timing = {
                    "ttft": first[0],
                    "generation": duration,
                    "tokens_in": length,
                    "tokens_out": len(ids),
                    "actual_batch_size": len(fresh),
                    "device": "cpu",
                    "precision": "float32",
                    "cache_hit": False,
                    "request_hash": key,
                }
                self.calls += 1
                self.input_tokens += length
                self.output_tokens += len(ids)
                (self.cache / (key + ".json")).write_text(
                    json.dumps(
                        {"response": response, "response_hash": digest(response), "timing": timing}
                    )
                )
                for pos, index in enumerate(pending[key]["indices"]):
                    shared = dict(timing)
                    if pos:
                        shared = {
                            "cache_hit": True,
                            "tokens_in": 0,
                            "tokens_out": 0,
                            "generation": 0,
                            "original": timing,
                            "request_hash": key,
                            "same_batch_deduplicated": True,
                        }
                    output[index] = (response, shared)
            self.batch_records.append(
                {
                    "batch_size": len(fresh),
                    "wall_seconds": duration,
                    "padded_input_length": inputs["input_ids"].shape[1],
                }
            )
        return output


class ClientBackend:
    def __init__(self, client):
        self.client = client

    def generate(self, payload, max_tokens):
        response = self.client.generate(SYSTEM, json.dumps(payload), max_tokens, 11)
        return response, dict(self.client.timings[-1])


def run(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "runner.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    tasks, manifest = load_public(args.manifest)
    private = load_scorer(args.manifest)
    protocol = json.loads(Path("configs/v3_coding/protocol.json").read_text())
    by_id = {t.identity: t for t in tasks}
    evaluation = [by_id[key] for key in protocol["evaluation_task_ids"]]
    identity = {
        "manifest": manifest["public_hash"],
        "protocol": digest(protocol),
        "source_hash": source_hash(),
        "runner_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "backbone_files": hashes(args.model),
        "training_files": hashes(args.training),
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": "cpu",
        "precision": "float32",
        "conditions": args.conditions,
        "prompts": "unchanged frozen-v1; no truncation",
    }
    ipath = root / "identity.json"
    if ipath.exists() and json.loads(ipath.read_text()) != identity:
        raise ValueError("Resume identity changed")
    ipath.write_text(json.dumps(identity, indent=2))
    executor = Executor(args.runtime)
    training = [t for t in tasks if t.split == "train"]
    memory = VerifiedMemory.load(Path(args.training) / "memory.json", training, executor, private)
    ledger = json.loads((Path(args.training) / "ledger.json").read_text())
    if ledger["source_hash"] != source_hash():
        raise ValueError("Frozen training source changed")
    checkpoint = Path(args.training) / f"evaluator-s{args.seed}.pt"
    expected = next(r["checkpoint_sha256"] for r in ledger["receipts"] if r["seed"] == args.seed)
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != expected:
        raise ValueError("Checkpoint changed")
    state = torch.load(checkpoint, weights_only=True)
    model = ProposalEvaluator(args.seed)
    model.load_state_dict(state["state"])
    model.eval()
    jobs = []
    completed = 0
    for task in evaluation:
        for condition in args.conditions:
            path = root / f"{task.identity}-{condition}.json"
            if path.exists():
                saved = json.loads(path.read_text())
                if saved["run_identity"] != digest(identity) or saved["trace_hash"] != digest(
                    saved["trace"]
                ):
                    raise ValueError("Saved trace changed")
                completed += 1
            else:
                jobs.append((task, condition))
    started = time.perf_counter()
    coder = CpuBatchCoder(args.model, args.cache) if jobs else None

    def episode(job, client):
        task, condition = job
        trace = solve(
            task,
            ClientBackend(client),
            executor,
            memory,
            model,
            condition,
            protocol["max_rounds"],
            protocol["max_tokens"],
        )
        terminal = executor.run(trace["repository"], private[task.identity]["private_tests"])
        first = (
            executor.run(trace["snapshots"][0], private[task.identity]["private_tests"])
            if trace["snapshots"]
            else None
        )
        trace.update(
            success=terminal["all_passed"],
            terminal_private=terminal,
            first_edit_success=first["all_passed"] if first else False,
            private_vm_calls=int(terminal["actual_vm_call"])
            + int(first["actual_vm_call"] if first else False),
        )
        return trace

    def save(trace):
        nonlocal completed
        path = root / f"{trace['task_id']}-{trace['condition']}.json"
        path.write_text(
            json.dumps(
                {"run_identity": digest(identity), "trace_hash": digest(trace), "trace": trace},
                indent=2,
            )
        )
        completed += 1
        print(
            json.dumps(
                {
                    "complete": completed,
                    "condition": trace["condition"],
                    "family": trace["family"],
                    "success": trace["success"],
                }
            ),
            flush=True,
        )

    try:
        if jobs:
            evaluate_batch(
                coder,
                jobs,
                episode,
                save,
                batch_size=args.batch_size,
                deadline=time.time() + args.seconds,
            )
    finally:
        if (
            hashes(args.model) != identity["backbone_files"]
            or hashes(args.training) != identity["training_files"]
        ):
            raise ValueError("Immutable inputs changed")
        status = {
            "completed": completed,
            "expected": len(evaluation) * len(args.conditions),
            "complete": completed == len(evaluation) * len(args.conditions),
            "elapsed_seconds": time.perf_counter() - started,
            "cap_seconds": args.seconds,
            "model_calls": coder.calls if coder else 0,
            "tokens_in": coder.input_tokens if coder else 0,
            "tokens_out": coder.output_tokens if coder else 0,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "batches": coder.batch_records if coder else [],
            "lockbox_accessed": False,
            "backbone_and_training_unchanged": True,
            "numerical_stratum": "CPU float32 left-padded batch up to " + str(args.batch_size),
        }
        (root / f"invocation-{time.time_ns()}.json").write_text(json.dumps(status, indent=2))
        print(json.dumps({k: v for k, v in status.items() if k != "batches"}), flush=True)
    return status


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--training", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--runtime", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--cache", default="results/v3_coding/batched-cache")
    p.add_argument("--seconds", type=int, default=1800)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    run(p.parse_args())
