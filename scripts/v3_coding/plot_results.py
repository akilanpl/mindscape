"""Static research figure: primary-seed task accuracy and task-level Wilson intervals."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mindscape.v3.coding_learning.agent import CONDITIONS


def plot(summary, output):
    data = json.loads(Path(summary).read_text())
    cells = {(c["condition"], c["split"]): c for c in data["cells"] if c["seed"] == 11}
    x = np.arange(len(CONDITIONS))
    figure, axis = plt.subplots(figsize=(11, 4.8), layout="constrained")
    for offset, split, color in ((-0.18, "iid", "#8eb8d4"), (0.18, "ood", "#365d8d")):
        values = np.asarray([cells[c, split]["accuracy"] for c in CONDITIONS])
        intervals = np.asarray([cells[c, split]["wilson95"] for c in CONDITIONS])
        axis.bar(
            x + offset,
            values,
            width=0.34,
            color=color,
            label=f"{split.upper()} (n={6 if split == 'iid' else 12})",
        )
        axis.errorbar(
            x + offset,
            values,
            yerr=np.stack((values - intervals[:, 0], intervals[:, 1] - values)),
            fmt="none",
            color="#263342",
            linewidth=1,
            capsize=3,
        )
    axis.axhline(0.6, color="#a65832", linestyle="--", linewidth=1, label="Predeclared OOD target")
    axis.set_xticks(x, [c.replace("_", "\n") for c in CONDITIONS], fontsize=9)
    axis.set_ylim(0, 1.04)
    axis.set_ylabel("Terminal private-test task accuracy")
    axis.set_title("Mindscape V3: fresh Python repair panel, primary evaluator seed11", fontsize=12)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False, fontsize=9)
    axis.spines[["top", "right"]].set_visible(False)
    figure.text(
        0.5,
        -0.04,
        "Bars: 18 fixed tasks, not432 independent samples. Error bars: descriptive Wilson95%. Frozen greedy Qwen backbone.",
        ha="center",
        fontsize=8,
    )
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--summary", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    plot(a.summary, a.output)
