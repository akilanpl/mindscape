"""One-shot bounded vertical lockbox; never used to choose a configuration."""

import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path

from mindscape.coding.generator import OOD, TRAIN, task

root = Path("results/coding/completion_lockbox_v1")
root.mkdir(parents=True, exist_ok=True)
if (root / "tasks.json").exists():
    raise RuntimeError("Lockbox already frozen")
dataset = json.loads(Path("results/coding/final_dataset_v1/dataset.json").read_text())
excluded = {t["metadata"]["seed"] for ts in dataset.values() for t in ts}
rng = random.Random(13001)
tasks = []
for split, categories in [("test", TRAIN), ("ood_test", OOD)]:
    for i in range(50):
        while True:
            seed = rng.randrange(100000, 999999999)
            if seed not in excluded:
                break
        excluded.add(seed)
        tasks.append(asdict(task(seed, categories[i % len(categories)], split)))
blob = json.dumps(tasks, sort_keys=True)
(root / "tasks.json").write_text(blob)
(root / "manifest.json").write_text(
    json.dumps(
        {
            "tasks": 100,
            "iid": 50,
            "ood": 50,
            "sha256": hashlib.sha256(blob.encode()).hexdigest(),
            "selection_rule": "Preselected 1.5B backbone, budget100, training seed11; full typed-action system, at most three edits/eight actions; all four frozen controls evaluated once on identical tasks",
            "seed": 13001,
            "used_for_tuning": False,
        },
        indent=2,
    )
)
print("Frozen 100-task lockbox")
