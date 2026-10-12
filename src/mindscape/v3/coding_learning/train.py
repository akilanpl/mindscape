"""Train-only executed proposal evidence and small learned ranking checkpoints."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch

from mindscape.v3.schema import digest

from .data import load_public, load_scorer
from .execution import Executor
from .memory import VerifiedMemory
from .planning import features, fit
from .schema import RepairProposal


def train(root, output, runtime, seeds=(11, 23, 37)):
    output = Path(output)
    if output.exists():
        raise ValueError("Training destination must be new")
    tasks, manifest = load_public(root)
    private = load_scorer(root)
    tasks = [t for t in tasks if t.split == "train"]
    executor = Executor(runtime)
    memory = VerifiedMemory(tasks)
    rows = []
    events = []
    for task in tasks:
        correct = private[task.identity]["teacher_repository"]
        proposals = [RepairProposal(correct), RepairProposal(task.repository)]
        # Actual training negatives: invalid source and semantically wrong return.
        entry = task.contract.entry.rsplit(".", 1)[0].replace(".", "/") + ".py"
        proposals += [
            RepairProposal({entry: "def broken(:\n"}),
            RepairProposal(
                {entry: "def " + task.contract.entry.rsplit(".", 1)[1] + "(*args):\n    return 0\n"}
            ),
        ]
        for p in proposals:
            repo = p.apply(task.repository)
            observed = executor.run(repo, task.visible_tests)
            seconds = observed["execution"]["duration"] if observed["execution"] else 0
            rows.append(
                {
                    "split": "train",
                    "task_id": task.identity,
                    "features": features(task.repository, p, task.contract, ()).tolist(),
                    "visible_pass": observed["all_passed"],
                    "seconds": seconds,
                }
            )
            events.append(
                {"task_id": task.identity, "proposal": asdict(p), "actual_visible_result": observed}
            )
        memory.admit(task, correct, executor, private[task.identity]["private_tests"])
    # Retrieval support is computed leave-task-out, never from the query's own repair.
    for row, event in zip(rows, events):
        task = next(t for t in tasks if t.identity == row["task_id"])
        retrieved = [
            r
            for r in memory.retrieve(task.repository, task.contract, k=len(tasks))
            if r["task_id"] != task.identity
        ][:2]
        row["features"] = features(
            task.repository, RepairProposal(**event["proposal"]), task.contract, retrieved
        ).tolist()
    output.mkdir(parents=True)
    memory.save(output / "memory.json")
    (output / "training-evidence.json").write_text(
        json.dumps({"rows": rows, "events": events}, indent=2)
    )
    receipts = []
    for seed in seeds:
        model, receipt = fit(rows, seed)
        path = output / f"evaluator-s{seed}.pt"
        torch.save(
            {
                "state": model.state_dict(),
                "seed": seed,
                "manifest": manifest["public_hash"],
                "training_task_ids": [t.identity for t in tasks],
            },
            path,
        )
        receipt["checkpoint_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        receipts.append(receipt)
    ledger = {
        "manifest": manifest["public_hash"],
        "training_task_ids": [t.identity for t in tasks],
        "distinct_training_examples": len(tasks),
        "teacher_successes": len(memory.records),
        "proposal_labels": len(rows),
        "actual_vm_calls": sum(e["actual_visible_result"]["actual_vm_call"] for e in events)
        + 2 * len(tasks),
        "source_hash": source_hash(),
        "receipts": receipts,
        "model_training_tokens": 0,
        "qwen_updated": False,
        "environment": "training-only expert and negative candidate validation",
    }
    (output / "ledger.json").write_text(json.dumps(ledger, indent=2))
    return ledger


def source_hash():
    root = Path(__file__).parent
    return digest(
        {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("*.py"))}
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--runtime", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    a = p.parse_args()
    print(json.dumps(train(a.manifest, a.output, a.runtime, a.seeds)))
