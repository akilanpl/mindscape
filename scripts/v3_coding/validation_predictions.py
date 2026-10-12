"""Post-freeze validation-only scorer diagnostic; never updates any weights."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mindscape.v3.coding_learning.data import load_public, load_scorer
from mindscape.v3.coding_learning.execution import Executor
from mindscape.v3.coding_learning.memory import VerifiedMemory
from mindscape.v3.coding_learning.planning import ProposalEvaluator, features
from mindscape.v3.coding_learning.schema import RepairProposal


def near_miss(task, reference):
    source = reference["solution.py"]
    if task.family == "threshold":
        source = source.replace(">=", ">")
    elif task.family == "exclusive":
        source = source.replace(", n))", ", n+1))")
    elif task.family == "mutation":
        source = source.replace("sorted(values)", "sorted(values, reverse=True)")
    elif task.family == "ordering":
        source = source.replace("result.append(value)", "result.insert(0, value)")
    elif task.family == "accumulator":
        source = source.replace("result = 1", "result = 2")
    elif task.family == "empty":
        source = source.replace("default=", "default=1 + ")
    else:
        raise ValueError("Unknown validation family")
    return reference | {"solution.py": source}


def run(args):
    tasks, _ = load_public(args.manifest)
    scorer = load_scorer(args.manifest)
    executor = Executor(args.runtime)
    memory = VerifiedMemory.load(
        Path(args.training) / "memory.json",
        [t for t in tasks if t.split == "train"],
        executor,
        scorer,
    )
    models = {}
    for seed in (11, 23, 37):
        model = ProposalEvaluator(seed)
        model.load_state_dict(
            torch.load(Path(args.training) / f"evaluator-s{seed}.pt", weights_only=True)["state"]
        )
        models[seed] = model
    rows = []
    for task in tasks:
        if task.split != "validation":
            continue
        reference = scorer[task.identity]["teacher_repository"]
        retrieved = memory.retrieve(task.repository, task.contract)
        for kind, repository in (
            ("teacher", reference),
            ("unchanged", task.repository),
            ("near_miss", near_miss(task, reference)),
        ):
            proposal = RepairProposal(repository)
            actual = executor.run(repository, task.visible_tests)
            vector = features(task.repository, proposal, task.contract, retrieved)
            rows.append(
                {
                    "task_id": task.identity,
                    "family": task.family,
                    "kind": kind,
                    "repository": repository,
                    "actual_visible": actual,
                    "features": vector.tolist(),
                    "predictions": {seed: model.predict(vector) for seed, model in models.items()},
                }
            )
    summaries = []
    for seed in models:
        pairs = [
            (r["predictions"][seed]["visible_usefulness"], float(r["actual_visible"]["all_passed"]))
            for r in rows
        ]
        pos = [p for p, y in pairs if y]
        neg = [p for p, y in pairs if not y]
        auc = np.mean([float(p > n) + 0.5 * float(p == n) for p in pos for n in neg])
        summaries.append(
            {
                "seed": seed,
                "rows": len(pairs),
                "positive_rows": len(pos),
                "brier": float(np.mean([(p - y) ** 2 for p, y in pairs])),
                "accuracy_at_half": float(np.mean([(p >= 0.5) == bool(y) for p, y in pairs])),
                "auc": float(auc),
                "cost_mae_seconds": float(
                    np.mean(
                        [
                            abs(
                                r["predictions"][seed]["seconds"]
                                - (
                                    r["actual_visible"]["execution"]["duration"]
                                    if r["actual_visible"]["execution"]
                                    else 0
                                )
                            )
                            for r in rows
                        ]
                    )
                ),
            }
        )
    Path(args.output).write_text(
        json.dumps(
            {
                "scope": "post-freeze validation diagnosis; no weight updates or holdout tuning",
                "unique_validation_tasks": 6,
                "summaries": summaries,
                "sandbox_requests_including_memory_revalidation": executor.calls,
                "qwen_calls": 0,
                "rows": rows,
            },
            indent=2,
        )
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="configs/v3_coding/benchmark_v2")
    p.add_argument("--runtime", required=True)
    p.add_argument("--training", required=True)
    p.add_argument("--output", required=True)
    run(p.parse_args())
