"""Paired task-cluster intervals and complete failures, including false visible repairs."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from mindscape.v3.experiment import wilson


def analyze(root, output):
    rows = [
        json.loads(p.read_text())["trace"]
        for p in Path(root).glob("*-*.json")
        if p.name[:1] in "0123456789abcdef" and len(p.name.split("-")[0]) == 64
    ]
    cells = []
    groups = defaultdict(list)
    paired = defaultdict(dict)
    for r in rows:
        groups[r["condition"], r["split"]].append(r)
        paired[r["task_id"]][r["condition"]] = r
    for (condition, split), items in sorted(groups.items()):
        k = sum(r["success"] for r in items)
        n = len(items)
        preds = []
        for r in items:
            for a in r["attempts"]:
                if a.get("actual_visible_result") and a["selected"] is not None:
                    preds.append(
                        (
                            a["predictions"][a["selected"]]["visible_usefulness"],
                            float(a["actual_visible_result"]["all_passed"]),
                        )
                    )
        cells.append(
            {
                "condition": condition,
                "split": split,
                "successes": k,
                "total": n,
                "accuracy": k / n,
                "wilson95": wilson(k, n),
                "first_edit_successes": sum(r["first_edit_success"] for r in items),
                "self_corrected": sum(r["success"] and not r["first_edit_success"] for r in items),
                "false_visible_repairs": sum(
                    not r["success"]
                    and any(
                        a.get("actual_visible_result", {}).get("all_passed") for a in r["attempts"]
                    )
                    for r in items
                ),
                "parse_failures": sum(bool(a["parse_error"]) for r in items for a in r["attempts"]),
                "model_calls": sum(r["model_calls"] for r in items),
                "visible_vm_calls": sum(r["visible_vm_calls"] for r in items),
                "private_vm_calls": sum(r["private_vm_calls"] for r in items),
                "tokens_in_new": sum(
                    a["timing"].get("tokens_in", 0) for r in items for a in r["attempts"]
                ),
                "tokens_out_new": sum(
                    a["timing"].get("tokens_out", 0) for r in items for a in r["attempts"]
                ),
                "wall_seconds_including_cached_requests": sum(r["wall_seconds"] for r in items),
                "visible_brier": float(np.mean([(p - y) ** 2 for p, y in preds]))
                if preds
                else None,
                "calibration_target": "visible execution only, not private success",
                "failed_task_ids": [r["task_id"] for r in items if not r["success"]],
            }
        )
    rng = np.random.default_rng(840014)
    comparisons = []
    for split in ("iid", "ood"):
        for baseline in (
            "plain",
            "structured",
            "typed",
            "no_memory",
            "no_planning",
            "no_concepts",
            "word_memory",
        ):
            pairs = [
                (v["mindscape"], v[baseline])
                for v in paired.values()
                if "mindscape" in v and baseline in v and v["mindscape"]["split"] == split
            ]
            if not pairs:
                continue
            diffs = np.asarray([int(a["success"]) - int(b["success"]) for a, b in pairs])
            samples = np.mean(diffs[rng.integers(0, len(diffs), (2000, len(diffs)))], axis=1)
            comparisons.append(
                {
                    "split": split,
                    "baseline": baseline,
                    "paired_tasks": len(pairs),
                    "delta": float(np.mean(diffs)),
                    "paired_bootstrap95": list(np.quantile(samples, [0.025, 0.975])),
                    "wins": int(sum(diffs > 0)),
                    "losses": int(sum(diffs < 0)),
                }
            )
    result = {
        "cells": cells,
        "paired": comparisons,
        "episodes": len(rows),
        "limits": "Small fixed synthetic panel, descriptive unadjusted intervals; no seed-population or superiority inference. Cache-shared compute is not cold latency.",
    }
    Path(output).write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    print(json.dumps(analyze(a.run, a.output)))
