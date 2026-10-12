"""Task-cluster paired intervals, exact paired tests and complete error totals."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from mindscape.v3.experiment import wilson


def mcnemar_exact(wins, losses):
    n = wins + losses
    if not n:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def summarize(root, output):
    root, output = Path(root), Path(output)
    identity = json.loads((root / "identity.json").read_text())
    cells, by_condition, compute = [], {}, []
    for path in sorted(root.glob("eval-*/receipt.json")) + sorted(
        root.glob("control-*/receipt.json")
    ):
        receipt = json.loads(path.read_text())
        eid = receipt["identity"]
        condition = eid["condition"]
        budget = eid.get("budget", 0)
        seed = eid["seed"]
        by_condition[condition, budget, seed] = receipt["rows"]
        groups = defaultdict(list)
        for row in receipt["rows"]:
            groups[row["split"], row["family"]].append(row)
            if row["split"] == "ood":
                groups["ood", "all"].append(row)
        for (split, family), rows in groups.items():
            k, n = sum(r["success"] for r in rows), len(rows)
            errors = {
                key: sum(r["errors"].get(key, 0) for r in rows)
                for key in (
                    "operation",
                    "raw",
                    "digit",
                    "carry",
                    "next_state",
                    "consistency_false_accept",
                    "consistency_false_reject",
                )
            }
            steps = sum(r["steps"] for r in rows)
            cells.append(
                {
                    "condition": condition,
                    "budget": budget,
                    "seed": seed,
                    "split": split,
                    "family": family,
                    "successes": k,
                    "total": n,
                    "accuracy": k / n,
                    "wilson_95": wilson(k, n),
                    "strict_rollout_successes": sum(bool(r["strict_rollout"]) for r in rows),
                    "steps": steps,
                    "errors": errors,
                    "next_state_accuracy": 1 - errors["next_state"] / steps if steps else None,
                }
            )
        compute.append(
            {
                "condition": condition,
                "budget": budget,
                "seed": seed,
                **{
                    k: receipt[k]
                    for k in (
                        "elapsed_seconds",
                        "cpu_seconds",
                        "forward_calls",
                        "memory_hits",
                        "plan_candidates",
                    )
                },
            }
        )
    paired = []
    rng = np.random.default_rng(20261013)
    for budget in identity["budgets"]:
        for baseline in ("procedural", "no_memory", "no_planning"):
            differences = []
            allwins = alllosses = 0
            per_seed = []
            for seed in identity["seeds"]:
                left = {
                    r["task_id"]: r
                    for r in by_condition.get(("concepts", budget, seed), [])
                    if r["split"] == "ood"
                }
                right = {
                    r["task_id"]: r
                    for r in by_condition.get((baseline, budget, seed), [])
                    if r["split"] == "ood"
                }
                if not left or set(left) != set(right):
                    continue
                keys = sorted(left)
                diff = [int(left[k]["success"]) - int(right[k]["success"]) for k in keys]
                differences.append(diff)
                wins = sum(x == 1 for x in diff)
                losses = sum(x == -1 for x in diff)
                allwins += wins
                alllosses += losses
                per_seed.append(
                    {
                        "seed": seed,
                        "wins": wins,
                        "losses": losses,
                        "mcnemar_exact_p": mcnemar_exact(wins, losses),
                    }
                )
            if not differences:
                continue
            # Resample tasks, carrying every seed's paired difference with each task.
            array = np.asarray(differences)
            indices = rng.integers(array.shape[1], size=(2000, array.shape[1]))
            boot = array[:, indices].mean(axis=(0, 2))
            paired.append(
                {
                    "budget": budget,
                    "baseline": baseline,
                    "mean_difference": float(array.mean()),
                    "task_cluster_bootstrap_95": np.quantile(boot, [0.025, 0.975]).tolist(),
                    "per_seed": per_seed,
                    "unit": "tasks conditional on these trained seeds",
                }
            )
    # Predeclared Holm family: all per-seed OOD paired tests across 7 budgets x3 baselines x3 seeds.
    tests = [p for comparison in paired for p in comparison["per_seed"]]
    ordered = sorted(tests, key=lambda t: t["mcnemar_exact_p"])
    previous = 0.0
    for i, test in enumerate(ordered):
        previous = max(previous, min(1.0, test["mcnemar_exact_p"] * (len(tests) - i)))
        test["holm_p"] = previous
    curves = []
    for condition in sorted({k[0] for k in by_condition}):
        for budget in identity["budgets"]:
            for family in (
                "all",
                "length",
                "carry_chain",
                "digit_composition",
                "sign_change",
                "composition",
            ):
                values = [
                    c["accuracy"]
                    for c in cells
                    if c["condition"] == condition
                    and c["budget"] == budget
                    and c["split"] == "ood"
                    and c["family"] == family
                ]
                if values:
                    curves.append(
                        {
                            "condition": condition,
                            "budget": budget,
                            "family": family,
                            "mean": float(np.mean(values)),
                            "seed_sample_sd": float(np.std(values, ddof=1))
                            if len(values) > 1
                            else None,
                            "seeds": len(values),
                        }
                    )
    thresholds = []
    for condition in sorted({c["condition"] for c in curves}):
        for family in ("all", "length"):
            reached = [
                c["budget"]
                for c in curves
                if c["condition"] == condition and c["family"] == family and c["mean"] >= 0.5
            ]
            thresholds.append(
                {
                    "condition": condition,
                    "family": family,
                    "threshold": 0.5,
                    "first_observed_budget": min(reached) if reached else "not_reached",
                }
            )
    result = {
        "identity": identity,
        "cells": cells,
        "learning_curves": curves,
        "paired": paired,
        "holm_family_size": len(tests),
        "thresholds": thresholds,
        "evaluation_compute": compute,
        "lockbox_status": "encrypted_unopened",
        "interval_warning": "Conditional task sampling; degenerate intervals do not establish population equivalence.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    result = summarize(a.run, a.output)
    print(json.dumps({"cells": len(result["cells"]), "paired_comparisons": len(result["paired"])}))


if __name__ == "__main__":
    main()
