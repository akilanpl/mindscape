"""Measured coding summaries; missing jobs fail closed before final publication."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from mindscape.coding.statistics import bootstrap_mean, der, log_aulc, paired_delta, threshold

p = argparse.ArgumentParser()
p.add_argument("--output", default="results/coding/completion_analysis_v1")
a = p.parse_args()
root = Path(a.output)
root.mkdir(parents=True, exist_ok=True)
required = (
    "completion_gradient_v1",
    "completion_lockbox_eval_v1",
    "zero_budget_v1",
    "stress_v1",
    "latency_probe_v1",
)
for name in required:
    if not Path("results/coding", name, "complete.json").exists():
        raise RuntimeError("Incomplete study: " + name)


def read(name):
    return [
        json.loads(s) for s in Path("results/coding", name, "rows.jsonl").read_text().splitlines()
    ]


learning = read("completion_gradient_v1") + read("zero_budget_v1")
curves = defaultdict(dict)
cells = defaultdict(list)
for row in learning:
    cells[(row["condition"], row["split"], row["budget"])].append(row)
records = []
for (condition, split, budget), rows in sorted(cells.items()):
    seeds = defaultdict(list)
    tasks = defaultdict(list)
    for r in rows:
        seeds[r["seed"]].append(float(r["success"]))
        tasks[r["task_id"]].append(float(r["success"]))
    means = [float(np.mean(v)) for v in seeds.values()]
    score = float(np.mean(means))
    curves[(condition, split)][budget] = score
    records.append(
        {
            "condition": condition,
            "split": split,
            "budget": budget,
            "accuracy": score,
            "seed_means": means,
            "seed_sd": float(np.std(means, ddof=1)) if len(means) > 1 else None,
            "task_cluster_ci": bootstrap_mean([np.mean(v) for v in tasks.values()]),
            "episodes": len(rows),
            "zero_control_not_independent_seeds": budget == 0,
        }
    )
thresholds = {
    condition + "|" + split: {str(q): threshold(curve, q) for q in (0.8, 0.9, 0.95)}
    for (condition, split), curve in curves.items()
}
ratios = {
    condition + "|" + split: {
        str(q): der(threshold(curves[("model_only", split)], q), threshold(curve, q))
        for q in (0.8, 0.9, 0.95)
    }
    for (condition, split), curve in curves.items()
}
paired = []
for split in ("test", "ood_test"):
    for budget in (10, 25, 50, 100):
        baseline = {
            r["task_id"]: np.mean(
                [
                    float(x["success"])
                    for x in cells[("model_only", split, budget)]
                    if x["task_id"] == r["task_id"]
                ]
            )
            for r in cells[("model_only", split, budget)]
        }
        for condition in ("structured", "mindscape_b", "mindscape_c"):
            other = {
                r["task_id"]: np.mean(
                    [
                        float(x["success"])
                        for x in cells[(condition, split, budget)]
                        if x["task_id"] == r["task_id"]
                    ]
                )
                for r in cells[(condition, split, budget)]
            }
            paired.append(
                {
                    "condition": condition,
                    "split": split,
                    "budget": budget,
                    "paired_task_delta": paired_delta(baseline, other),
                }
            )
final = read("completion_lockbox_eval_v1")
final_cells = defaultdict(list)
for r in final:
    final_cells[r["condition"]].append(r)
final_summary = {}
for condition, rows in final_cells.items():
    valid = lambda r: (
        not any(not t["valid"] for t in r["trajectory"]["transitions"])
        and not any(a.get("parse_error") for a in r.get("attempts", []))
    )
    final_summary[condition] = {
        "success": bootstrap_mean([r["success"] for r in rows]),
        "successes": sum(r["success"] for r in rows),
        "grounded_success": bootstrap_mean([r["success"] and valid(r) for r in rows]),
        "valid_episode_rate": np.mean([valid(r) for r in rows]),
        "success_at_1": np.mean(
            [r.get("first_patch_success", r["success"]) or False for r in rows]
        ),
        "interactive_success_at_3": np.mean([r["success"] for r in rows]),
        "public_pass_k": False,
        "split_accuracy": {
            s: np.mean([r["success"] for r in rows if r["split"] == s])
            for s in {r["split"] for r in rows}
        },
    }
latency = {}
for condition in final_cells:
    rows = [r for r in read("latency_probe_v1") if r["condition"] == condition]
    values = {"wall_seconds": [r["wall_seconds"] for r in rows]}
    for r in rows:
        for stage, measurement in r["stages"].items():
            values.setdefault(stage, []).append(measurement["seconds"])
        for attempt in r.get("attempts", []):
            timing = attempt.get("latency", {})
            if timing.get("ttft") is not None:
                values.setdefault("ttft", []).append(timing["ttft"])
    latency[condition] = {
        name: {"n": len(v), "p50_p95_p99": np.quantile(v, [0.5, 0.95, 0.99]).tolist()}
        for name, v in values.items()
        if v
    }
stress = defaultdict(list)
for r in read("stress_v1"):
    stress[(r["kind"], str(r["variant"]))].append(r)
summary = {
    "learning_cells": records,
    "thresholds": thresholds,
    "DER": ratios,
    "normalized_log_AULC": {c + "|" + s: log_aulc(v) for (c, s), v in curves.items()},
    "paired_deltas": paired,
    "final": final_summary,
    "latency_fresh": latency,
    "stress": {
        k + "|" + v: {
            "n": len(rows),
            "successes": sum(r["success"] for r in rows),
            "accuracy": np.mean([r["success"] for r in rows]),
        }
        for (k, v), rows in stress.items()
    },
    "limitations": [
        "IID/OOD learning pools are diagnostic and previously exposed; final100 is independent",
        "Matched unique training tasks do not match action supervision, tokens, feedback or compute",
        "Task-cluster confidence intervals average observed training seeds; not population-wide seed uncertainty",
        "Unreached thresholds are right censored; 0/0 DER is undefined",
        "P99 from fourteen episodes per condition is exploratory",
        "Inference ablations retain trained weights; no causal retraining claim",
    ],
}
(root / "metrics.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary["final"], indent=2))
