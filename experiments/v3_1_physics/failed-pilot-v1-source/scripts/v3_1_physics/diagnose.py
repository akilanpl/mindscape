"""Training/development-only diagnosis. Native MPS is required for old model queries."""

import json
from pathlib import Path

import numpy as np
import torch

from mindscape.v3.physics_world.accelerator import require_mps
from mindscape.v3.physics_world.data import load
from mindscape.v3.physics_world.evaluation import metrics, public_batch, rollout
from mindscape.v3.physics_world.model import WorldModel
from mindscape.v3.physics_world.storage import atomic_json
from mindscape.v3.physics_world.training import restore


def run(output):
    require_mps()
    dataset = Path("results/v3_physics/dataset-full-v1")
    manifest = load(dataset)
    results = []
    for family in ("motion", "interaction"):
        for seed in (11, 23):
            for variant in (
                "mindscape",
                "no_concepts",
                "no_memory",
                "no_history",
                "no_relationships",
            ):
                root = (
                    Path("results/v3_physics/native-full-1791811860380119000/study")
                    / family
                    / f"s{seed}/n250"
                    / variant
                )
                identity = json.loads((root / "checkpoint-index.json").read_text())["identity"]
                model = WorldModel(variant, seed)
                optimizer = torch.optim.Adam(model.parameters())
                restore(root / "checkpoint-index.json", model, optimizer, identity)
                rows = []
                for r in [
                    r
                    for r in manifest["records"]
                    if r["scene"]["family"] == family and r["scene"]["split"] == "validation"
                ][:6]:
                    with np.load(dataset / r["file"], allow_pickle=False) as d:
                        rows.append({**{k: d[k] for k in d.files}, "horizon": 32})
                for horizon in (8, 16, 32):
                    for row in rows:
                        row["horizon"] = horizon
                    predicted = rollout(model, public_batch(rows))
                    values = [
                        metrics({k: v[i] for k, v in predicted.items()}, row, np.ones(4))
                        for i, row in enumerate(rows)
                    ]
                    results.append(
                        {
                            "family": family,
                            "seed": seed,
                            "variant": variant,
                            "horizon": horizon,
                            "development_rollout_mse": float(
                                np.mean([v["rollout_mse"] for v in values])
                            ),
                            "largest_predicted_absolute": float(abs(predicted["states"]).max()),
                        }
                    )
                del model, optimizer
    atomic_json(
        output,
        {
            "scope": "V3 validation only; never old IID/OOD",
            "rows": results,
            "device": "mps",
            "old_final_test_reads": 0,
        },
    )
