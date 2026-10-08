"""Reconstruct historical generation prefixes without running/altering old studies."""

import argparse
import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path

from mindscape.coding.generator import TRAIN, generate, task

p = argparse.ArgumentParser()
p.add_argument("--output", default="results/coding")
a = p.parse_args()
root = Path(a.output)
dataset = {
    k: [asdict(t) for t in values]
    for k, values in generate(
        counts={"train": 1000, "validation": 20, "test": 20, "ood_test": 20}
    ).items()
}
excluded = {t["metadata"]["seed"] for values in dataset.values() for t in values}
rng = random.Random(7002)
while len(dataset["train"]) < 5000:
    seed = rng.randrange(100000, 999999999)
    if seed in excluded:
        continue
    excluded.add(seed)
    dataset["train"].append(asdict(task(seed, TRAIN[len(dataset["train"]) % len(TRAIN)], "train")))
folder = root / "study_v2"
folder.mkdir(parents=True, exist_ok=True)
path = folder / "dataset.json"
blob = json.dumps(dataset, sort_keys=True)
if path.exists():
    if json.loads(path.read_text()) != dataset:
        raise RuntimeError("Existing dataset differs; never overwrite")
else:
    path.write_text(blob)
print("Historical dataset source reconstructed", hashlib.sha256(blob.encode()).hexdigest())
