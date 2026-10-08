"""Evidence summaries; missing cells never become fabricated zeroes."""

import argparse
import json
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("root")
a = p.parse_args()
root = Path(a.root)
rows = [json.loads(x) for x in (root / "rows.jsonl").read_text().splitlines()]
rng = np.random.default_rng(991)


def percentiles(xs):
    return {str(q): float(np.percentile(xs, q)) for q in (50, 95, 99)} if xs else None


cells = []
for key in sorted({(r["condition"], r["budget"], r["split"]) for r in rows}):
    group = [r for r in rows if (r["condition"], r["budget"], r["split"]) == key]
    values = np.array([r["success"] for r in group], float)
    boot = rng.choice(values, size=(2000, len(values)), replace=True).mean(1)
    first = [r["attempts"][0]["latency"]["ttft"] for r in group]
    calls = [attempt["latency"]["generation"] for r in group for attempt in r["attempts"]]
    steps = [len(r["trajectory"]["transitions"]) for r in group]
    tools = [
        t["result"]["duration"]
        for r in group
        for t in r["trajectory"]["transitions"]
        if t["action"]["name"].startswith("run_test")
    ]
    recovery = [r for r in group if len(r["attempts"]) > 1]
    cells.append(
        {
            "condition": key[0],
            "budget": key[1],
            "split": key[2],
            "n": len(group),
            "success_rate": float(values.mean()),
            "bootstrap_task_ci95": np.percentile(boot, [2.5, 97.5]).tolist(),
            "ttft_seconds": percentiles(first),
            "generation_seconds": percentiles(calls),
            "wall_seconds": percentiles([r["wall_seconds"] for r in group]),
            "execution_seconds": percentiles(tools),
            "steps": percentiles(steps),
            "recovery_cases": len(recovery),
            "recovery_success_rate": sum(r["success"] for r in recovery) / len(recovery)
            if recovery
            else None,
            "grounded_tool_result_rate": sum(
                not t["result"]["hypothetical"] and t["result"]["evidence_kind"] == "actual_result"
                for r in group
                for t in r["trajectory"]["transitions"]
            )
            / sum(steps),
            "trajectory_valid_rate": sum(
                t["valid"] for r in group for t in r["trajectory"]["transitions"]
            )
            / sum(steps),
            "tokens_in": sum(att["latency"]["tokens_in"] for r in group for att in r["attempts"]),
            "tokens_out": sum(att["latency"]["tokens_out"] for r in group for att in r["attempts"]),
        }
    )
thresholds = []
for condition in sorted({c["condition"] for c in cells}):
    for split in ("test", "ood_test"):
        for threshold in (0.8, 0.9, 0.95):
            eligible = [
                c["budget"]
                for c in cells
                if c["condition"] == condition
                and c["split"] == split
                and c["success_rate"] >= threshold
            ]
            thresholds.append(
                {
                    "condition": condition,
                    "split": split,
                    "threshold": threshold,
                    "N_star_retrieval": min(eligible) if eligible else None,
                    "gradient_DER": "not estimable from retrieval study",
                }
            )
report = {
    "cells": cells,
    "thresholds": thresholds,
    "bootstrap_limitation": "Task resampling pooled across retrieval seeds; correlated shared tasks; not independent training-seed significance",
    "claims": {
        "real_execution": "SUPPORTED",
        "host_capability_isolation": "SUPPORTED by probes, not a formal proof",
        "gradient_data_efficiency": "NOT SUPPORTED",
        "general_software_engineering": "NOT SUPPORTED",
        "GPT_Gemini_equivalence": "NOT SUPPORTED",
    },
    "completed_rows": len(rows),
}
(root / "analysis.json").write_text(json.dumps(report, indent=2))
print(len(rows), len(cells))
