"""Scene-paired statistics and completeness; frames never count as independent episodes."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from mindscape.v3.experiment import wilson

from .schema import digest
from .storage import atomic_json


def analyze(root, output):
    root = Path(root)
    identity = json.loads((root / "identity.json").read_text())
    groups = defaultdict(list)
    planning = defaultdict(list)
    traces = []
    for path in root.rglob("*.json"):
        if len(path.stem) != 64:
            continue
        saved = json.loads(path.read_text())
        trace = saved["trace"]
        if digest(trace) != saved["trace_hash"]:
            raise ValueError("Evidence changed")
        info = saved["identity"]
        key = (
            info["family"],
            info.get("seed", 0),
            info.get("budget", 0),
            info["variant"],
            trace["split"],
        )
        if "metrics" in trace:
            groups[key].append(trace)
            traces.append((key, trace))
        else:
            planning[key].append(trace)
    rng = np.random.default_rng(20261012)
    cells = []
    for key, rows in sorted(groups.items()):
        values = np.asarray([r["metrics"]["normalized_rollout_mse"] for r in rows])
        samples = values[rng.integers(0, len(values), (2000, len(values)))].mean(1)
        cells.append(
            {
                "family": key[0],
                "seed": key[1],
                "budget": key[2],
                "variant": key[3],
                "split": key[4],
                "scenes": len(rows),
                "normalized_rollout_error": float(values.mean()),
                "scene_bootstrap95": np.quantile(samples, [0.025, 0.975]).tolist(),
                "one_step_mse": float(np.mean([r["metrics"]["one_step_mse"] for r in rows])),
                "finite_scenes": sum(r["metrics"]["finite"] for r in rows),
            }
        )
    comparisons = []
    for key, rows in groups.items():
        if key[3] != "mindscape":
            continue
        baseline = groups.get((*key[:3], "matched", key[4]))
        if not baseline:
            continue
        base = {r["scene_id"]: r for r in baseline}
        pairs = [(r, base[r["scene_id"]]) for r in rows if r["scene_id"] in base]
        if len(pairs) != len(rows):
            raise ValueError("Unpaired comparison")
        a = np.asarray([r["metrics"]["normalized_rollout_mse"] for r, _ in pairs])
        b = np.asarray([r["metrics"]["normalized_rollout_mse"] for _, r in pairs])
        index = rng.integers(0, len(a), (2000, len(a)))
        ratio = 1 - a[index].mean(1) / np.maximum(b[index].mean(1), 1e-12)
        comparisons.append(
            {
                "family": key[0],
                "seed": key[1],
                "budget": key[2],
                "split": key[4],
                "paired_scenes": len(pairs),
                "relative_error_reduction": float(1 - a.mean() / max(b.mean(), 1e-12)),
                "paired_bootstrap95": np.quantile(ratio, [0.025, 0.975]).tolist(),
            }
        )
    plans = [
        {
            "family": k[0],
            "seed": k[1],
            "budget": k[2],
            "variant": k[3],
            "split": k[4],
            "scenes": len(rows),
            "successes": sum(r["goal_success"] for r in rows),
            "wilson95": wilson(sum(r["goal_success"] for r in rows), len(rows)),
            "actual_environment_steps": sum(r["actual_environment_steps"] for r in rows),
        }
        for k, rows in sorted(planning.items())
    ]
    primary = sum(len(v) for k, v in groups.items() if k[3] == "mindscape")
    required = identity["primary_required"]
    result = {
        "schema": "mindscape.v3.physics.analysis/1",
        "primary_completed": primary,
        "primary_required": required,
        "primary_complete": primary == required,
        "cells": cells,
        "paired": comparisons,
        "planning": plans,
        "real_world_validation": "NOT RUN",
        "model_free_neural_control": "See planning cells; receives additional teacher action-choice labels, not equal-information comparator",
        "limits": "Fixed scene-level bootstrap, unadjusted multiple comparisons. Seeds share scenes; no frame or seed pooling as independent samples. GPU evidence required separately.",
    }
    atomic_json(output, result)
    return result
