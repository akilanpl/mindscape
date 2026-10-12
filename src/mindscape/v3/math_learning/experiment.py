"""Fixed protocol, complete traces and curves; never reads the encrypted lockbox/key."""

import argparse
import gzip
import json
import os
import platform
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from mindscape.v3.schema import canonical, digest

from .data import load
from .environment import LocalState, expected, run_problem, teacher_local
from .learner import Predictor, TransitionNet
from .training import build_training, file_hash, train


def source_hash():
    return digest({p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))})


@contextmanager
def lock(root):
    path = root / ".lock"
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        yield
    finally:
        os.close(fd)
        path.unlink()


class Oracle:
    """Explicit upper bound; never used for neural training/evaluation selection."""

    forward_calls = 0

    def predict_local(self, state):
        return teacher_local(state), {"consistency": 1.0, "oracle": True}

    def predict_sign(self, visible):
        from .environment import Problem

        return int(
            expected(
                Problem(tuple(visible["operands"]), tuple(visible["operations"]), "unit", "oracle")
            )
            < 0
        ), {"oracle": True}


def evaluate(problems, predictor, directory, mode, identity):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    complete = root / "receipt.json"
    if complete.exists():
        result = json.loads(complete.read_text())
        if result["identity"] != identity:
            raise ValueError("Evaluation identity changed")
        for name, sha in result["files"].items():
            if file_hash(root / name) != sha:
                raise ValueError("Completed evaluation corrupt")
        return result["rows"]
    began, cpu = time.perf_counter(), time.process_time()
    rows = []
    tmp = root / "traces.tmp.gz"
    with gzip.open(tmp, "wt") as stream:
        for problem in problems:
            started = time.perf_counter()
            if mode == "answer_only":
                value, logits = predictor.answer(problem.visible())
                trace = {
                    "task_id": problem.identity,
                    "problem": asdict(problem),
                    "answer": str(value),
                    "expected": str(expected(problem)),
                    "success": value == expected(problem),
                    "strict_rollout": None,
                    "steps": 0,
                    "events": [],
                    "errors": {},
                    "answer_logits": logits,
                }
            else:
                trace = run_problem(problem, predictor)
            trace["latency_seconds"] = time.perf_counter() - started
            # Scores are retained once per unique observed context in contexts.json.gz.
            for event in trace["events"]:
                if "diagnostics" in event:
                    event["diagnostics"] = {
                        k: v for k, v in event["diagnostics"].items() if k != "logits"
                    }
                    event["context_key"] = canonical(LocalState(**event["state"]).key)
            stream.write(canonical(trace) + "\n")
            rows.append(
                {
                    k: trace[k]
                    for k in (
                        "task_id",
                        "success",
                        "strict_rollout",
                        "steps",
                        "errors",
                        "latency_seconds",
                    )
                }
            )
            rows[-1].update(split=problem.split, family=problem.family)
    tmp.replace(root / "traces.jsonl.gz")
    contexts = getattr(predictor, "cache", {})
    with gzip.open(root / "contexts.json.gz", "wt") as stream:
        json.dump(
            {
                canonical(key): {"claim": asdict(value[0]), "diagnostics": value[1]}
                for key, value in contexts.items()
            },
            stream,
        )
    result = {
        "identity": identity,
        "rows": rows,
        "files": {name: file_hash(root / name) for name in ("traces.jsonl.gz", "contexts.json.gz")},
        "elapsed_seconds": time.perf_counter() - began,
        "cpu_seconds": time.process_time() - cpu,
        "forward_calls": getattr(predictor, "forward_calls", 0),
        "memory_hits": getattr(predictor, "memory_hits", 0),
        "plan_candidates": getattr(predictor, "plan_candidates", 0),
        "cache_scope": "read-only neural predictions for identical observable local states",
    }
    (root / "receipt.json").write_text(json.dumps(result, indent=2))
    return rows


def run(
    manifest_root,
    output,
    budgets=None,
    seeds=None,
    updates=600,
    pretrained_path=None,
    modes=("procedural", "concepts", "no_memory", "no_planning", "answer_only"),
):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    frozen, tasks = load(manifest_root)
    budgets = budgets or frozen["body"]["budgets"]
    seeds = seeds or frozen["body"]["seeds"]
    if any(b not in frozen["body"]["budgets"] for b in budgets):
        raise ValueError("Unfrozen budget")
    if any(s not in frozen["body"]["seeds"] for s in seeds):
        raise ValueError("Unfrozen seed")
    if any(
        mode not in ("procedural", "concepts", "no_memory", "no_planning", "answer_only")
        for mode in modes
    ):
        raise ValueError("Unknown evaluation condition")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    encoder = None
    if pretrained_path:
        from .pretrained import FrozenEncoder

        encoder = FrozenEncoder(pretrained_path, root / "frozen-features.npz")
    identity = {
        "schema": "v3.math.run/1",
        "manifest_hash": frozen["sha256"],
        "source_hash": source_hash(),
        "budgets": list(budgets),
        "seeds": list(seeds),
        "updates": updates,
        "modes": list(modes),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "pretrained_hashes": encoder.hashes if encoder else None,
    }
    with lock(root):
        if (root / "identity.json").exists():
            if json.loads((root / "identity.json").read_text()) != identity:
                raise ValueError("Run resume identity mismatch")
        else:
            (root / "identity.json").write_text(json.dumps(identity, indent=2))
        evaluation = tasks["validation"] + tasks["iid"] + tasks["ood"]
        for budget in budgets:
            arrays, memory, ledger, demos = build_training(tasks["train"][:budget])
            demo_path = root / f"demonstrations-{budget}.jsonl.gz"
            if not demo_path.exists():
                with gzip.open(demo_path, "wt") as stream:
                    for demo in demos:
                        stream.write(canonical(demo) + "\n")
            for seed in seeds:
                for regime in ("procedural", "answer_only"):
                    if regime == "answer_only" and "answer_only" not in modes:
                        continue
                    model, receipt = train(
                        arrays,
                        ledger,
                        root / f"model-{regime}-n{budget}-s{seed}",
                        seed,
                        updates,
                        regime,
                        encoder,
                        identity["source_hash"],
                        identity["manifest_hash"],
                    )
                    conditions = (
                        ("answer_only",)
                        if regime == "answer_only"
                        else tuple(m for m in modes if m != "answer_only")
                    )
                    for mode in conditions:
                        predictor = Predictor(
                            model,
                            memory if mode in ("concepts", "no_planning") else {},
                            mode in ("concepts", "no_memory"),
                            encoder,
                        )
                        eid = {
                            "run_hash": digest(identity),
                            "checkpoint": receipt["checkpoint_sha256"],
                            "budget": budget,
                            "seed": seed,
                            "condition": mode,
                        }
                        evaluate(
                            evaluation,
                            predictor,
                            root / f"eval-{mode}-n{budget}-s{seed}",
                            mode,
                            eid,
                        )
                print(canonical({"budget": budget, "seed": seed, "complete": True}), flush=True)
            if encoder:
                encoder.save()
        # Controls are fixed once, not misleading copies at each data budget.
        for condition in ("random", "zero", "oracle"):
            for seed in seeds if condition == "random" else (0,):
                if condition == "oracle":
                    predictor = Oracle()
                else:
                    model = TransitionNet(seed, encoder.size if encoder else 0)
                    if condition == "zero":
                        for parameter in model.parameters():
                            parameter.data.zero_()
                    predictor = Predictor(model, {}, False, encoder)
                evaluate(
                    evaluation,
                    predictor,
                    root / f"control-{condition}-s{seed}",
                    condition,
                    {"run_hash": digest(identity), "condition": condition, "seed": seed},
                )
        if encoder:
            encoder.save()
            (root / "encoder-compute.json").write_text(
                json.dumps(
                    {
                        "forward_batches": encoder.calls,
                        "processed_tokens": encoder.tokens,
                        "requests": encoder.requests,
                        "frozen_parameters": encoder.parameter_count,
                        "cache_entries": len(encoder.values),
                        "scope": "shared cache across all conditions; inference-only encoder, no label adaptation",
                    },
                    indent=2,
                )
            )
    return root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budgets", type=int, nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--updates", type=int, default=600)
    parser.add_argument("--pretrained-path")
    parser.add_argument("--modes", nargs="+")
    args = parser.parse_args()
    run(
        args.manifest,
        args.output,
        args.budgets,
        args.seeds,
        args.updates,
        args.pretrained_path,
        tuple(args.modes)
        if args.modes
        else ("procedural", "concepts", "no_memory", "no_planning", "answer_only"),
    )


if __name__ == "__main__":
    main()
