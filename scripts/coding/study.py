"""Frozen, resumable real-execution retrieval adaptation study (not fine-tuning)."""

import argparse
import hashlib
import json
import random
import resource
import statistics
from dataclasses import asdict
from pathlib import Path

from mindscape.coding.generator import generate
from mindscape.coding.memory import CodingMemory
from mindscape.coding.model import LocalCoder
from mindscape.coding.policy import CodingPolicy, teacher_memory
from mindscape.coding.sandbox import WasiSandbox

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--train", type=int, default=1000)
p.add_argument("--locked-from")
p.add_argument("--teacher-from")
p.add_argument("--collect-only", action="store_true")
p.add_argument("--test", type=int, default=20)
p.add_argument("--budgets", default="10,25,50,100,250,500,1000")
p.add_argument("--seeds", default="11,23,37")
p.add_argument("--output", default="results/coding/study_v1")
a = p.parse_args()
root = Path(a.output)
root.mkdir(parents=True, exist_ok=True)
budgets = [int(x) for x in a.budgets.split(",")]
seeds = [int(x) for x in a.seeds.split(",")]
splits = generate(counts={"train": a.train, "validation": 20, "test": a.test, "ood_test": a.test})
if a.locked_from:
    from mindscape.coding.generator import TRAIN, task
    from mindscape.coding.schema import CodeTask

    locked = json.loads(Path(a.locked_from).read_text())
    for split in ("validation", "test", "ood_test"):
        splits[split] = [CodeTask(**t) for t in locked[split]]
    splits["train"] = [CodeTask(**t) for t in locked["train"]]
    excluded = {t.metadata["seed"] for ts in splits.values() for t in ts}
    rng = random.Random(7002)
    while len(splits["train"]) < a.train:
        seed = rng.randrange(100000, 999999999)
        if seed in excluded:
            continue
        excluded.add(seed)
        splits["train"].append(task(seed, TRAIN[len(splits["train"]) % len(TRAIN)], "train"))
sandbox = WasiSandbox("work/coding/runtime")
sandbox.probe()
manifest = {split: [asdict(t) for t in tasks] for split, tasks in splits.items()}
blob = json.dumps(manifest, sort_keys=True)
(root / "dataset.json").write_text(blob)
hashes = {
    split: {
        hashlib.sha256(json.dumps(t.repository, sort_keys=True).encode()).hexdigest() for t in ts
    }
    for split, ts in splits.items()
}
audit = {
    "exact_repository_overlap": {
        f"{x}:{y}": len(hashes[x] & hashes[y]) for x in hashes for y in hashes if x < y
    },
    "iid_shared_templates": "intentional",
    "ood_structures_disjoint": True,
    "hidden_tests_returned_to_policy": False,
    "dataset_sha256": hashlib.sha256(blob.encode()).hexdigest(),
}
(root / "leakage.json").write_text(json.dumps(audit, indent=2))
config = {
    "adaptation": "nearest-example retrieval of executed training trajectories; no gradient updates",
    "budgets": budgets,
    "seeds": seeds,
    "test_per_split": a.test,
    "conditions": ["model_only", "structured", "trajectory", "experiential"],
    "max_attempts": 3,
    "model_revision": Path(a.model).name,
    "dataset_sha256": audit["dataset_sha256"],
    "sampling": "greedy",
    "max_tool_steps": 8,
    "source_sha256": hashlib.sha256(
        "".join(p.read_text() for p in sorted(Path("src/mindscape/coding").glob("*.py"))).encode()
    ).hexdigest(),
    "seeds_change": "training memory subsets/order; generation deterministic",
}
existing = root / "protocol.json"
if existing.exists() and json.loads(existing.read_text()) != config:
    raise RuntimeError("Frozen protocol mismatch")
existing.write_text(json.dumps(config, indent=2))
cache = root / "teacher_memory.json"
if cache.exists():
    records = json.loads(cache.read_text())
elif a.teacher_from:
    records = json.loads(Path(a.teacher_from).read_text())
    cache.write_text(json.dumps(records))
else:
    records = []
    for index, t in enumerate(splits["train"]):
        m = teacher_memory([t], sandbox, 1)
        records.extend(m.episodic)
        if (index + 1) % 50 == 0:
            cache.write_text(json.dumps(records))
            print("executed training trajectories", index + 1, flush=True)
    cache.write_text(json.dumps(records))
# Resume an interrupted teacher pass without relabeling or imagined outcomes.
if len(records) != a.train:
    for t in splits["train"][len(records) :]:
        records.extend(teacher_memory([t], sandbox, 1).episodic)
        if len(records) % 50 == 0:
            cache.write_text(json.dumps(records))
            print("executed training trajectories", len(records), flush=True)
    cache.write_text(json.dumps(records))
if a.collect_only:
    (root / "collection_complete.json").write_text(
        json.dumps(
            {
                "executed_teacher_examples": len(records),
                "dataset_sha256": audit["dataset_sha256"],
                "mode": "actual tuple collection; model evaluation deferred",
            }
        )
    )
    raise SystemExit(0)
coder = LocalCoder(a.model)
rows_path = root / "rows.jsonl"
rows = [json.loads(x) for x in rows_path.read_text().splitlines()] if rows_path.exists() else []
done = {tuple(r["key"]) for r in rows}
for seed in seeds:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    for budget in budgets:
        if budget > a.train:
            continue
        for condition in config["conditions"]:
            # Frozen controls have no sample-budget-dependent parameter or memory state.
            if condition in ("model_only", "structured") and (
                budget != budgets[0] or seed != seeds[0]
            ):
                continue
            memory = CodingMemory(episodic=shuffled[:budget])
            policy = CodingPolicy(coder, sandbox, condition, memory)
            for split in ("test", "ood_test"):
                for t in splits[split]:
                    key = (seed, budget, condition, split, t.task_id)
                    if key in done:
                        continue
                    row = policy.solve(t, seed)
                    row.update(key=key, seed=seed, budget=budget, split=split)
                    with rows_path.open("a") as f:
                        f.write(json.dumps(row) + "\n")
                    rows.append(row)
                    done.add(key)
                selected = [
                    r
                    for r in rows
                    if r["seed"] == seed
                    and r["budget"] == budget
                    and r["condition"] == condition
                    and r["split"] == split
                ]
                print(
                    seed,
                    budget,
                    condition,
                    split,
                    sum(r["success"] for r in selected),
                    len(selected),
                    flush=True,
                )
summary = {
    "protocol": config,
    "parameters": coder.parameter_count,
    "completed_rows": len(rows),
    "cells": [],
    "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
}
for condition in config["conditions"]:
    for budget in [budgets[0]] if condition in ("model_only", "structured") else budgets:
        for split in ("test", "ood_test"):
            group = [
                r
                for r in rows
                if r["condition"] == condition and r["budget"] == budget and r["split"] == split
            ]
            if group:
                summary["cells"].append(
                    {
                        "condition": condition,
                        "budget": budget,
                        "split": split,
                        "n": len(group),
                        "success_rate": sum(r["success"] for r in group) / len(group),
                        "mean_wall_seconds": statistics.mean(r["wall_seconds"] for r in group),
                        "calls": sum(r["model_calls"] for r in group),
                    }
                )
(root / "summary.json").write_text(json.dumps(summary, indent=2))
