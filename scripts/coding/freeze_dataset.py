"""Freeze unseen primary cases after explicit diagnostic parser development."""

import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path

from mindscape.coding.generator import OOD, TRAIN, task

root = Path("results/coding/final_dataset_v1")
root.mkdir(parents=True, exist_ok=True)
if (root / "dataset.json").exists():
    raise RuntimeError("Dataset already frozen")
source = Path("results/coding/study_v2/dataset.json")
dataset = json.loads(source.read_text())
excluded = {t["metadata"]["seed"] for ts in dataset.values() for t in ts}
rng = random.Random(9001)
for split, categories in [("test", TRAIN), ("ood_test", OOD)]:
    values = []
    for i in range(20):
        while True:
            seed = rng.randrange(100000, 999999999)
            if seed not in excluded:
                break
        excluded.add(seed)
        values.append(asdict(task(seed, categories[i % len(categories)], split)))
    dataset[split] = values
blob = json.dumps(dataset, sort_keys=True)
(root / "dataset.json").write_text(blob)
manifest = {
    "dataset_sha256": hashlib.sha256(blob.encode()).hexdigest(),
    "train": 5000,
    "test": 20,
    "ood_test": 20,
    "fresh_heldout_seed": 9001,
    "training_preserved": True,
    "prior_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "diagnostic_sets": "study_v1/study_v2 held-out cases were exposed by adapter format diagnostics and are excluded from primary accuracy claims",
    "changes_since_diagnostics": "generic source-edit envelope normalization; no source-code changes, no model tuning from hidden outcomes",
}
(root / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(manifest)
