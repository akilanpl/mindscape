"""Frozen offline Qwen repair comparison with resume and explicit resource receipts."""

import argparse
import fcntl
import hashlib
import json
import time
from pathlib import Path

import torch

from mindscape.v3.schema import digest

from .agent import CONDITIONS, solve
from .data import load_public, load_scorer
from .execution import Executor
from .memory import VerifiedMemory
from .planning import ProposalEvaluator
from .train import source_hash


def hashes(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(root).rglob("*"))
        if p.is_file()
    }


class QwenBackend:
    def __init__(self, path, cache):
        from mindscape.coding.model import LocalCoder

        self.coder = LocalCoder(path, cache_enabled=False, device="cpu", precision="float32")
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.model_hash = digest(hashes(path))

    def generate(self, payload, max_tokens):
        system = "You are a Python debugging researcher. Return exactly the requested JSON. Use only visible evidence."
        key = digest([self.model_hash, system, payload, max_tokens, "cpu", "float32", "greedy"])
        path = self.cache / (key + ".json")
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["key"] != key or digest(saved["response"]) != saved["response_hash"]:
                raise ValueError("Generation cache changed")
            return saved["response"], {
                "cache_hit": True,
                "tokens_in": 0,
                "tokens_out": 0,
                "generation": 0,
                "original": saved["timing"],
                "request_hash": key,
            }
        response = self.coder.generate(system, json.dumps(payload), max_tokens, 11)
        timing = dict(self.coder.timings[-1])
        timing.update(cache_hit=False, request_hash=key)
        path.write_text(
            json.dumps(
                {
                    "key": key,
                    "response": response,
                    "response_hash": digest(response),
                    "timing": timing,
                }
            )
        )
        return response, timing


def run(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "runner.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    tasks, manifest = load_public(args.manifest)
    private = load_scorer(args.manifest)
    protocol = json.loads(Path("configs/v3_coding/protocol.json").read_text())
    selected = {x for x in protocol["evaluation_task_ids"]}
    by_id = {t.identity: t for t in tasks}
    evaluation = [by_id[key] for key in protocol["evaluation_task_ids"]]
    if len(selected) != len(evaluation) or any(t.split not in ("iid", "ood") for t in evaluation):
        raise ValueError("Invalid evaluation panel")
    training = [t for t in tasks if t.split == "train"]
    identity = {
        "manifest": manifest["public_hash"],
        "protocol": digest(protocol),
        "source_hash": source_hash(),
        "backbone_files": hashes(args.model),
        "training_files": hashes(args.training),
        "conditions": args.conditions,
        "seed": args.seed,
        "device": "cpu",
        "precision": "float32",
        "max_tokens": protocol["max_tokens"],
        "max_rounds": protocol["max_rounds"],
    }
    receipt = root / "identity.json"
    if receipt.exists() and json.loads(receipt.read_text()) != identity:
        raise ValueError("Resume identity changed")
    receipt.write_text(json.dumps(identity, indent=2))
    executor = Executor(args.runtime)
    memory = VerifiedMemory.load(Path(args.training) / "memory.json", training, executor, private)
    ledger = json.loads((Path(args.training) / "ledger.json").read_text())
    if ledger["source_hash"] != source_hash():
        raise ValueError("Training source changed")
    checkpoint = Path(args.training) / f"evaluator-s{args.seed}.pt"
    expected = next(r["checkpoint_sha256"] for r in ledger["receipts"] if r["seed"] == args.seed)
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != expected:
        raise ValueError("Evaluator checkpoint changed")
    saved = torch.load(checkpoint, weights_only=True)
    if saved["manifest"] != manifest["public_hash"]:
        raise ValueError("Wrong evaluator dataset")
    model = ProposalEvaluator(args.seed)
    model.load_state_dict(saved["state"])
    model.eval()
    completed = []
    pending = []
    for task in evaluation:
        for condition in args.conditions:
            path = root / f"{task.identity}-{condition}.json"
            if path.exists():
                r = json.loads(path.read_text())
                if r["run_identity"] != digest(identity) or r["trace_hash"] != digest(r["trace"]):
                    raise ValueError("Saved episode changed")
                completed.append(str(path))
            else:
                pending.append((task, condition, path))
    started = time.perf_counter()
    backend = QwenBackend(args.model, args.cache) if pending else None
    stopped = False
    try:
        for task, condition, path in pending:
            if time.perf_counter() - started >= args.seconds:
                stopped = True
                break
            trace = solve(
                task,
                backend,
                executor,
                memory,
                model,
                condition,
                protocol["max_rounds"],
                protocol["max_tokens"],
            )
            # Private scorer is called after solve returns, including all first-edit audits.
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
            path.write_text(
                json.dumps(
                    {"run_identity": digest(identity), "trace_hash": digest(trace), "trace": trace},
                    indent=2,
                )
            )
            completed.append(str(path))
            print(
                json.dumps(
                    {
                        "task": task.identity,
                        "condition": condition,
                        "success": trace["success"],
                        "complete": len(completed),
                    }
                ),
                flush=True,
            )
    finally:
        if (
            hashes(args.model) != identity["backbone_files"]
            or hashes(args.training) != identity["training_files"]
        ):
            raise ValueError("Immutable weights/training evidence changed")
        status = {
            "schema": "mindscape.v3.coding.run/1",
            "completed": len(completed),
            "expected": len(evaluation) * len(args.conditions),
            "elapsed_seconds": time.perf_counter() - started,
            "resource_cap_seconds": args.seconds,
            "resource_cap_reached": stopped,
            "complete": len(completed) == len(evaluation) * len(args.conditions),
            "model_calls": backend.coder.calls if backend else 0,
            "tokens_in": backend.coder.input_tokens if backend else 0,
            "tokens_out": backend.coder.output_tokens if backend else 0,
            "peak_rss_scope": "process high-water, not isolated per episode",
            "lockbox_accessed": False,
            "backbone_and_training_unchanged": True,
        }
        import resource

        status["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (root / f"invocation-{time.time_ns()}.json").write_text(json.dumps(status, indent=2))
        print(json.dumps(status), flush=True)
    return status


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--training", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--runtime", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--cache", default="results/v3_coding/generation-cache")
    p.add_argument("--seconds", type=int, default=1800)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    a = p.parse_args()
    run(a)
