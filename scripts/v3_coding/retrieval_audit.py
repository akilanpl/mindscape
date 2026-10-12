"""Post-freeze initial-state retrieval diagnosis using declared generator families."""

import argparse
import json
from pathlib import Path

from mindscape.v3.coding_learning.data import load_public, load_scorer
from mindscape.v3.coding_learning.execution import Executor
from mindscape.v3.coding_learning.memory import VerifiedMemory

EXPECTED = {
    "cross_file": {"threshold"},
    "compound": {"threshold", "accumulator"},
    "control_flow": {"exclusive"},
    "renaming": {"threshold"},
    "alias_mutation": {"mutation"},
    "composition": {"ordering"},
}


def run(args):
    tasks, _ = load_public(args.manifest)
    by_id = {t.identity: t for t in tasks}
    executor = Executor(args.runtime)
    memory = VerifiedMemory.load(
        Path(args.training) / "memory.json",
        [t for t in tasks if t.split == "train"],
        executor,
        load_scorer(args.manifest),
    )
    rows = []
    for task in tasks:
        if task.split not in ("iid", "ood"):
            continue
        expected = EXPECTED.get(task.family, {task.family})
        for strategy in ("structured", "words"):
            records = memory.retrieve(task.repository, task.contract, strategy=strategy)
            families = [by_id[r["task_id"]].family for r in records]
            rows.append(
                {
                    "task_id": task.identity,
                    "split": task.split,
                    "family": task.family,
                    "strategy": strategy,
                    "expected_mechanism_families": sorted(expected),
                    "retrieved_training_ids": [r["task_id"] for r in records],
                    "retrieved_families": families,
                    "some_expected_family": bool(expected.intersection(families)),
                    "all_expected_families": expected.issubset(families),
                }
            )
    summary = [
        {
            "split": split,
            "strategy": strategy,
            "some_expected": sum(
                r["some_expected_family"]
                for r in rows
                if r["split"] == split and r["strategy"] == strategy
            ),
            "all_expected": sum(
                r["all_expected_families"]
                for r in rows
                if r["split"] == split and r["strategy"] == strategy
            ),
            "total": sum(r["split"] == split and r["strategy"] == strategy for r in rows),
        }
        for split in ("iid", "ood")
        for strategy in ("structured", "words")
    ]
    Path(args.output).write_text(
        json.dumps(
            {
                "scope": "Post-freeze family-overlap diagnostic, not semantic retrieval truth or a training signal",
                "summary": summary,
                "rows": rows,
                "memory_revalidation_vm_requests": executor.calls,
                "model_calls": 0,
            },
            indent=2,
        )
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="configs/v3_coding/benchmark_v2")
    p.add_argument("--runtime", required=True)
    p.add_argument("--training", required=True)
    p.add_argument("--output", required=True)
    run(p.parse_args())
