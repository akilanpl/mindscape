"""Standalone scientific SVG figures from completed measured metrics."""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "work/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path("results/coding/completion_analysis_v1")
metrics = json.loads((root / "metrics.json").read_text())
plt.rcParams.update(
    {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"}
)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, split in zip(axes, ("test", "ood_test")):
    for condition in ("model_only", "structured", "mindscape_b", "mindscape_c"):
        rows = sorted(
            [
                r
                for r in metrics["learning_cells"]
                if r["split"] == split and r["condition"] == condition and r["budget"] > 0
            ],
            key=lambda r: r["budget"],
        )
        xs = [r["budget"] for r in rows]
        ys = [r["accuracy"] for r in rows]
        ax.plot(xs, ys, marker="o", label=condition)
        ax.fill_between(
            xs,
            [r["task_cluster_ci"]["ci95"][0] for r in rows],
            [r["task_cluster_ci"]["ci95"][1] for r in rows],
            alpha=0.1,
        )
    ax.set_xscale("log")
    ax.set_xticks([10, 25, 50, 100], labels=["10", "25", "50", "100"])
    ax.set_ylim(0, 1.02)
    ax.set_title("IID diagnostic" if split == "test" else "Structural OOD diagnostic")
    ax.set_xlabel("Unique gradient-training tasks")
    ax.grid(alpha=0.2)
axes[0].set_ylabel("All-hidden-tests success")
axes[1].legend(fontsize=8, loc="lower left")
fig.suptitle("Measured gradient curves · 3 adapter seeds · task-cluster 95% intervals")
fig.tight_layout()
fig.savefig(root / "learning_curves.svg")
plt.close(fig)
conditions = list(metrics["final"])
fig, ax = plt.subplots(figsize=(7, 3.6))
values = [metrics["final"][c]["success"]["mean"] for c in conditions]
errors = [
    [v - metrics["final"][c]["wilson_ci95"][0] for c, v in zip(conditions, values)],
    [metrics["final"][c]["wilson_ci95"][1] - v for c, v in zip(conditions, values)],
]
ax.bar(
    conditions, values, yerr=errors, capsize=4, color=["#6b8f9c", "#89a9b2", "#5f9f8a", "#2b7d68"]
)
ax.set_ylim(0, 1.02)
ax.set_ylabel("All-hidden-tests success")
ax.set_title("Independent frozen 100-task lockbox · Wilson 95% intervals")
ax.grid(axis="y", alpha=0.2)
fig.tight_layout()
fig.savefig(root / "locked_success.svg")
plt.close(fig)
print("Saved measured SVG figures")
