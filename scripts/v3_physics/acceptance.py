"""Independent exact-cell coverage and scene-cluster target checks; zero neural calls."""

import argparse
import json
from pathlib import Path

import numpy as np

from mindscape.v3.physics_world.data import BUDGETS, SEEDS, load
from mindscape.v3.physics_world.schema import digest
from mindscape.v3.physics_world.storage import atomic_json, file_sha


def check(dataset, root, output):
    manifest = load(dataset)
    root = Path(root)
    identity = json.loads((root / "identity.json").read_text())
    if identity["scope"] != "full_fixed":
        raise ValueError("Pilot is not full acceptance")
    rows = {}
    plans = {}
    for path in root.rglob("*.json"):
        if len(path.stem) != 64:
            continue
        saved = json.loads(path.read_text())
        trace = saved["trace"]
        info = saved["identity"]
        if saved["trace_hash"] != digest(trace):
            raise ValueError("Evidence changed")
        key = (
            info["family"],
            info.get("seed", 0),
            info.get("budget", 0),
            info["variant"],
            trace["split"],
            trace["scene_id"],
        )
        target = rows if "metrics" in trace else plans
        if key in target:
            raise ValueError("Duplicate scene-condition key")
        target[key] = trace
    missing = []
    expected = 0
    primary = 0
    matched = 0
    ablation = 0
    for family in ("motion", "interaction"):
        for split in ("iid", "ood"):
            scenes = [
                r["identity"]
                for r in manifest["records"]
                if r["scene"]["family"] == family and r["scene"]["split"] == split
            ]
            if len(scenes) != 100:
                raise ValueError("Fixed100-scene panel required")
            for seed in SEEDS:
                for budget in BUDGETS:
                    variants = ("mindscape", "matched") + (
                        (
                            "no_relationships",
                            "no_concepts",
                            "no_memory",
                            "no_history",
                            "no_feedback",
                        )
                        if budget == 250
                        else ()
                    )
                    for variant in variants:
                        for scene in scenes:
                            key = (family, seed, budget, variant, split, scene)
                            expected += 1
                            if key not in rows:
                                missing.append(list(key))
                                continue
                            if rows[key]["device"] != "mps":
                                raise ValueError("CPU neural evidence forbidden")
                            primary += variant == "mindscape"
                            matched += variant == "matched"
                            ablation += variant not in ("mindscape", "matched")
    control_missing = []
    baseline_count = 0
    for family in ("motion", "interaction"):
        for split in ("iid", "ood"):
            ids = [r["identity"] for r in manifest["records"]
                   if r["scene"]["family"] == family and r["scene"]["split"] == split]
            for variant in ("persistence", "constant_velocity", "analytical_available"):
                for scene in ids:
                    key = (family, 0, 0, variant, split, scene)
                    if key not in rows:
                        control_missing.append(list(key))
                    else:
                        baseline_count += 1
            for variant in ("random", "simple", "analytical"):
                for scene in ids[:20]:
                    key = (family, 0, 0, variant, split, scene)
                    if key not in plans:
                        control_missing.append(list(key))
    comparisons = []
    rng = np.random.default_rng(20261012)
    for family in ("motion", "interaction"):
        ids = [
            r["identity"]
            for r in manifest["records"]
            if r["scene"]["family"] == family and r["scene"]["split"] == "ood"
        ]
        if any(
            (family, seed, 250, variant, "ood", scene) not in rows
            for scene in ids
            for seed in SEEDS
            for variant in ("mindscape", "matched")
        ):
            continue
        a = np.asarray(
            [
                np.mean(
                    [
                        rows[family, seed, 250, "mindscape", "ood", scene]["metrics"][
                            "normalized_rollout_mse"
                        ]
                        for seed in SEEDS
                    ]
                )
                for scene in ids
            ]
        )
        b = np.asarray(
            [
                np.mean(
                    [
                        rows[family, seed, 250, "matched", "ood", scene]["metrics"][
                            "normalized_rollout_mse"
                        ]
                        for seed in SEEDS
                    ]
                )
                for scene in ids
            ]
        )
        indices = rng.integers(0, len(ids), (2000, len(ids)))
        reduction = 1 - a.mean() / max(b.mean(), 1e-12)
        samples = 1 - a[indices].mean(1) / np.maximum(b[indices].mean(1), 1e-12)
        comparisons.append(
            {
                "family": family,
                "independent_scene_clusters": len(ids),
                "seed_aggregation": "average within each scene, do not treat3 seeds as3 independent scenes",
                "relative_ood_error_reduction": float(reduction),
                "paired_scene_bootstrap95": np.quantile(samples, [0.025, 0.975]).tolist(),
                "target20_met": bool(reduction >= 0.2),
            }
        )
    plan_missing = []
    plan_differences = []
    for family in ("motion", "interaction"):
        for split in ("iid", "ood"):
            ids = [
                r["identity"]
                for r in manifest["records"]
                if r["scene"]["family"] == family and r["scene"]["split"] == split
            ][:20]
            for seed in SEEDS:
                for variant in (
                    "mindscape",
                    "matched",
                    "no_relationships",
                    "no_concepts",
                    "no_memory",
                    "no_history",
                    "no_feedback",
                    "model_free",
                ):
                    for scene in ids:
                        if (family, seed, 250, variant, split, scene) not in plans:
                            plan_missing.append([family, seed, variant, split, scene])
            if all(
                (family, seed, 250, variant, split, scene) in plans
                for scene in ids
                for seed in SEEDS
                for variant in ("mindscape", "matched")
            ):
                differences = np.asarray(
                    [
                        np.mean(
                            [
                                int(
                                    plans[family, seed, 250, "mindscape", split, scene][
                                        "goal_success"
                                    ]
                                )
                                - int(
                                    plans[family, seed, 250, "matched", split, scene][
                                        "goal_success"
                                    ]
                                )
                                for seed in SEEDS
                            ]
                        )
                        for scene in ids
                    ]
                )
                plan_differences.append(
                    {
                        "family": family,
                        "split": split,
                        "scene_clusters": 20,
                        "paired_success_delta": float(differences.mean()),
                        "maintained_or_improved": bool(differences.mean() >= 0),
                    }
                )
    result = {
        "primary_rollouts_completed": primary,
        "primary_required": 6000,
        "matched_rollouts": matched,
        "ablation_rollouts": ablation,
        "expected_neural_rollouts": expected,
        "missing_neural_cells": missing,
        "missing_planning_cells": plan_missing,
        "error_comparisons": comparisons,
        "planning_comparisons": plan_differences,
        "nonneural_forecast_rollouts": baseline_count,
        "missing_control_cells": control_missing,
        "coverage_complete": not missing and not plan_missing and not control_missing,
        "target_met": len(comparisons) == 2
        and all(r["target20_met"] for r in comparisons)
        and len(plan_differences) == 4
        and all(r["maintained_or_improved"] for r in plan_differences),
        "real_world_validation": "NOT RUN",
        "neural_calls": 0,
        "verification_receipt_required": True,
        "source_index": file_sha(root / "identity.json"),
    }
    atomic_json(output, result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("primary_rollouts_completed", "coverage_complete", "target_met")
            }
        )
    )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    check(a.dataset, a.run, a.output)
