"""Five actual training seeds; nested checkpoints, one pass per training task."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--samples", type=int, default=100)
p.add_argument("--seeds", default="11,23,37,53,71")
p.add_argument("--checkpoints", default="10,25,50,100")
p.add_argument("--output", default="results/coding/gradient_v1")
p.add_argument("--wait-for", action="append", default=[])
a = p.parse_args()
root = Path(a.output)
root.mkdir(parents=True, exist_ok=True)
# Prevent multiple resident training/inference models during the small initial jobs.
while any(not Path(path).exists() for path in a.wait_for):
    time.sleep(5)
for seed in [int(x) for x in a.seeds.split(",")]:
    folder = root / f"seed_{seed}"
    if (folder / "training.json").exists():
        continue
    subprocess.run(
        [
            sys.executable,
            "scripts/coding/train_lora.py",
            "--model",
            a.model,
            "--samples",
            str(a.samples),
            "--seed",
            str(seed),
            "--checkpoints",
            a.checkpoints,
            "--output",
            str(folder),
        ],
        check=True,
    )
manifest = {
    "seeds": [int(x) for x in a.seeds.split(",")],
    "samples": a.samples,
    "budgets": [int(x) for x in a.checkpoints.split(",")],
    "records": [
        json.loads((root / f"seed_{s}" / "training.json").read_text()) for s in a.seeds.split(",")
    ],
    "label": "real gradient adaptation; same patch-supervised coding backbone shared across architectural inference controls; retrieval study remains separate",
}
(root / "manifest.json").write_text(json.dumps(manifest, indent=2))
print("Gradient training completed", flush=True)
