"""Read training data and checkpoint bytes; no neural execution and no old final test reads."""

import json
from pathlib import Path

import numpy as np
import torch

from mindscape.v3.physics_world.storage import atomic_json

root = Path("results/v3_physics/dataset-full-v1")
manifest = json.loads((root / "manifest.json").read_text())
summary = {}
for family in ("motion", "interaction"):
    counts = []
    derivatives = []
    for r in manifest["records"]:
        if r["scene"]["family"] != family or r["scene"]["split"] != "train":
            continue
        with np.load(root / r["file"], allow_pickle=False) as d:
            counts.append(int(d["observations"][0, :, 9].sum()))
            derivatives.append(float(abs(np.diff(d["observations"][:, :, 2:4], axis=0) * 30).max()))
    checkpoint = Path(
        f"results/v3_physics/native-full-1791811860380119000/study/{family}/s11/n250/mindscape/checkpoint-index.json"
    )
    index = json.loads(checkpoint.read_text())
    saved = torch.load(checkpoint.parent / index["file"], weights_only=True, map_location="cpu")
    summary[family] = {
        "training_object_counts": sorted(set(counts)),
        "relation_feature_scales": saved["state"]["feature_scale"][88:94].tolist(),
        "largest_observed_acceleration_feature": max(derivatives),
    }
atomic_json(
    "experiments/v3_1_physics/training-diagnosis.json",
    {
        "scope": "training only",
        "neural_calls": 0,
        "old_final_test_reads": 0,
        "families": summary,
        "hypotheses": [
            "zero relational exposure in motion training",
            "unbounded recursive derivative feature amplification",
            "one-step/free-run distribution mismatch",
            "global flattened memory is slot-order sensitive",
        ],
    },
)
print(json.dumps(summary))
