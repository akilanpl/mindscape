"""CPU-only scientific plotting: simulated demonstrations or actual result curves."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("work/physics-matplotlib-cache").resolve()))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def demos(dataset, output):
    from mindscape.v3.physics_world.data import load
    from mindscape.v3.physics_world.schema import HISTORY

    manifest = load(dataset)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for axis, family in zip(axes, ("motion", "interaction"), strict=True):
        record = next(
            r
            for r in manifest["records"]
            if r["scene"]["family"] == family
            and r["scene"]["split"] == "train"
            and r["scene"]["index"] == 1
        )
        with np.load(Path(dataset) / record["file"], allow_pickle=False) as row:
            states = row["states"]
            for i in range(len(record["scene"]["objects"])):
                axis.plot(states[:, i, 0], states[:, i, 1], label=f"object{i}")
                axis.scatter(states[HISTORY - 1, i, 0], states[HISTORY - 1, i, 1], s=25)
        axis.axhline(0, color="gray", linewidth=1)
        axis.set_xlabel("Position x")
        axis.set_ylabel("Position y")
        axis.set_title(f"{family}: {record['scene']['mechanism']}")
        axis.legend(frameon=False)
    figure.suptitle("Pinned-engine simulation only — not neural predictions or real measurements")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def curves(analysis, output):
    data = json.loads(Path(analysis).read_text())
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for axis, family in zip(axes, ("motion", "interaction"), strict=True):
        for variant in ("mindscape", "matched"):
            cells = [
                c
                for c in data["cells"]
                if c["family"] == family and c["variant"] == variant and c["split"] == "ood"
            ]
            budgets = sorted({c["budget"] for c in cells})
            values = [
                np.mean([c["normalized_rollout_error"] for c in cells if c["budget"] == budget])
                for budget in budgets
            ]
            axis.plot(budgets, values, marker="o", label=variant)
        axis.set_title(family)
        axis.set_xlabel("Independent training trajectories")
        axis.set_ylabel("Mean normalized OOD rollout MSE")
        axis.set_yscale("log")
        axis.legend(frameon=False)
    figure.suptitle("Actual scene-level outcomes; seed means share evaluation scenes")
    figure.savefig(output, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset")
    p.add_argument("--analysis")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if bool(a.dataset) == bool(a.analysis):
        raise ValueError("Select exactly one input")
    demos(a.dataset, a.output) if a.dataset else curves(a.analysis, a.output)
