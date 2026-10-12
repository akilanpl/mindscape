"""Static scientific learning curves; all measured points and seed SD are retained."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(compact, adapter, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    studies = {
        "Compact neural": json.loads(Path(compact).read_text()),
        "MiniLM residual adapter": json.loads(Path(adapter).read_text()),
    }
    records = []
    for name, study in studies.items():
        for row in study["learning_curves"]:
            records.append({"model": name, **row})
    with (output / "learning-curves.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    colors = {
        "concepts": "#087e8b",
        "procedural": "#b44b43",
        "no_memory": "#707c91",
        "no_planning": "#9761a3",
        "answer_only": "#c39e38",
    }
    for col, (name, study) in enumerate(studies.items()):
        for row, family in enumerate(("all", "length")):
            ax = axes[row, col]
            for condition, color in colors.items():
                points = sorted(
                    [
                        r
                        for r in study["learning_curves"]
                        if r["family"] == family and r["condition"] == condition
                    ],
                    key=lambda r: r["budget"],
                )
                if not points:
                    continue
                xs = [p["budget"] for p in points]
                ys = [p["mean"] * 100 for p in points]
                sd = [p["seed_sample_sd"] * 100 for p in points]
                ax.errorbar(
                    xs,
                    ys,
                    yerr=sd,
                    marker="o",
                    markersize=3,
                    color=color,
                    label=condition.replace("_", " "),
                    linewidth=1.4,
                    capsize=2,
                )
            ax.axhline(50, color="#333333", linestyle="--", linewidth=0.8)
            ax.set_xscale("log")
            ax.set_xlim(8, 1200)
            ax.set_xticks(
                [10, 25, 50, 100, 250, 500, 1000],
                labels=["10", "25", "50", "100", "250", "500", "1000"],
            )
            ax.set_ylim(0, 100)
            ax.grid(alpha=0.18)
            ax.set_title(
                f"{name}: {'all structural OOD' if family == 'all' else 'predeclared length target'}"
            )
            ax.set_ylabel("Exact task accuracy (%)")
            ax.set_xlabel("Distinct training problems")
    axes[0, 0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Frozen V3 mathematics study — mean ± seed sample SD (3 seeds)", fontsize=13)
    fig.tight_layout()
    fig.savefig(output / "learning-curves.png", dpi=180)
    fig.savefig(output / "learning-curves.svg")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--compact", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    plot(a.compact, a.adapter, a.output)
