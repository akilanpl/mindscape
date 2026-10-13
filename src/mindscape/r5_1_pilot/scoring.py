"""Pure arithmetic scoring and frozen gates; no neural/executor imports."""

import math
import statistics

from mindscape.r5_1.metrics import aggregate, summarize


def public_valid(task):
    p = task["public"]
    if task["domain"] == "coding":
        return int(0 <= p["target_leaf"] < len(p["constants"]))
    size = len(p["values"]) if task["domain"] == "mathematics" else len(p["masses"])
    return int(0 <= p["destination"] < size)


def baseline_predictions(task):
    from mindscape.v5.data import graph

    g = graph(task)
    width = 4 if task["domain"] == "physics" else 1
    persistence = [[0.0] * width for _ in g["entities"]]
    local = [r.copy() for r in persistence]
    d = task["domain"]
    p = task["public"]
    valid = public_valid(task)
    if valid:
        if d == "mathematics":
            local[p["destination"]][0] = (1 if p["operation"] == "add" else -1) * p["amount"] / 100
        elif d == "coding":
            local[p["target_leaf"] + 1][0] = p["increment"] / 100
        else:
            local[p["destination"]][2:] = [v / 4 for v in p["impulse"]]
    return {
        key: {"identity": task["identity"], "delta": value, "valid_probability": float(valid)}
        for key, value in [("persistence", persistence), ("local_action", local)]
    }


def metrics(rows, predictions):
    if len(predictions) != len(rows) or len({p["identity"] for p in predictions}) != len(rows):
        raise ValueError("Prediction coverage/duplicates")
    by = {p["identity"]: p for p in predictions}
    records = []
    validity = {0: [], 1: []}
    brier = []
    invalid = []
    integer = []
    correct = []
    kinematic = []
    scalar = []
    details = []
    for row in rows:
        p = by[row["identity"]]
        target = row["target"]
        prob = p["valid_probability"]
        if not math.isfinite(prob) or not 0 <= prob <= 1:
            raise ValueError("Invalid probability")
        record = summarize(target, p["delta"])
        records.append(record)
        v = target["valid"]
        vcorrect = int((prob >= 0.5) == bool(v))
        validity[v].append(vcorrect)
        brier.append((prob - v) ** 2)
        values = [
            (x, y)
            for a, b in zip(p["delta"], target["delta"], strict=True)
            for x, y in zip(a, b, strict=True)
        ]
        scalar.extend((x - y) ** 2 for x, y in values)
        if not v:
            invalid.append(record["whole_graph_mse"])
        scale = 4 if row["domain"] == "physics" else 100
        if row["domain"] == "physics":
            success = all(abs(x - y) * scale <= 0.25 for x, y in values)
            if v:
                kinematic.extend(
                    abs(delta[i] * 4 - (before[i + 2] + delta[i + 2] * 4) / 30)
                    for before, delta in zip(target["before"], p["delta"], strict=True)
                    for i in (0, 1)
                )
        else:
            success = all(round(x * scale) == round(y * scale) for x, y in values)
            integer.extend(abs(x * 100 - round(x * 100)) <= 0.1 for x, y in values)
        correct.append(int(success and vcorrect))
        details.append(
            {
                "identity": row["identity"],
                **record,
                "validity_correct": vcorrect,
                "validity_brier": (prob - v) ** 2,
            }
        )
    result = aggregate(records)
    result.update(
        scalar_pooled_whole_graph_mse=statistics.mean(scalar),
        validity_balanced_accuracy=statistics.mean(
            statistics.mean(v) for v in validity.values() if v
        ),
        validity_brier=statistics.mean(brier),
        invalid_noop_mse=statistics.mean(invalid) if invalid else None,
        integer_consistency_fraction=statistics.mean(integer) if integer else None,
        transition_correct=sum(correct),
        transition_accuracy=statistics.mean(correct),
        physics_kinematic_error=statistics.mean(kinematic) if kinematic else None,
        finite=True,
    )
    return result, details


def mean(values):
    return statistics.mean(values)


def gate_report(fits, baseline_metrics, protocol):
    seeds = protocol["optimizer_seeds"]
    by = {
        (f["identity"]["target"], f["identity"]["condition"], f["identity"]["seed"]): f
        for f in fits
    }
    expected = {
        (d, c, s)
        for d in ("mathematics", "coding", "physics")
        for c in ("domain_only", "shared", "randomized_auxiliary")
        for s in seeds
    }
    if set(by) != expected or len(fits) != 27:
        raise ValueError("Incomplete fits")
    gains = {}
    for d in ("mathematics", "coding", "physics"):
        gains[d] = {}
        for c in ("domain_only", "randomized_auxiliary"):
            values = []
            for s in seeds:
                control = by[d, c, s]["composition"]["changed_entity_mse"]
                shared = by[d, "shared", s]["composition"]["changed_entity_mse"]
                values.append(
                    (control - shared) / control
                    if control is not None and control > 0 and shared is not None
                    else None
                )
            gains[d][c] = values
    g = protocol["scientific_gates_all_required"]
    shared = [by["coding", "shared", s] for s in seeds]
    ce = [f["composition"]["changed_entity_mse"] for f in shared]
    checks = {
        "primary_both_control_gains": all(
            all(x is not None for x in v)
            and mean(v) >= g["mean_relative_changed_mse_improvement_over_each_of_B_and_R"]
            for v in gains["coding"].values()
        ),
        "primary_seed_consistency": all(
            x is not None and x > 0 for v in gains["coding"].values() for x in v
        ),
        "persistence_improvement": all(x is not None for x in ce)
        and mean(ce)
        <= (1 - g["mean_shared_changed_mse_reduction_over_persistence"])
        * baseline_metrics["coding"]["persistence"]["changed_entity_mse"],
        "local_improvement": all(x is not None for x in ce)
        and mean(ce)
        <= (1 - g["mean_shared_changed_mse_reduction_over_local_action"])
        * baseline_metrics["coding"]["local_action"]["changed_entity_mse"],
        "validity": all(
            f["composition"]["validity_balanced_accuracy"]
            >= g["shared_validity_balanced_accuracy_every_seed"]
            for f in shared
        ),
        "whole_graph_guard": all(
            mean(f["composition"]["whole_graph_mse"] for f in shared)
            <= mean(by["coding", c, s]["composition"]["whole_graph_mse"] for s in seeds)
            for c in ("domain_only", "randomized_auxiliary")
        ),
        "unchanged_guard": all(f["composition"]["unchanged_entity_mse"] is not None for f in shared)
        and mean(f["composition"]["unchanged_entity_mse"] for f in shared)
        <= g["shared_mean_unchanged_entity_mse_maximum"],
        "invalid_guard": all(f["composition"]["invalid_noop_mse"] is not None for f in shared)
        and mean(f["composition"]["invalid_noop_mse"] for f in shared)
        <= g["shared_mean_invalid_noop_mse_maximum"],
        "other_domain_regression": all(
            all(x is not None for x in gains[d]["domain_only"])
            and mean(gains[d]["domain_only"])
            >= -g["maximum_mean_other_domain_relative_changed_mse_regression"]
            for d in ("mathematics", "physics")
        ),
        "learns_development": all(
            f["initial_training"]["changed_entity_mse"] > 0
            and f["final_training"]["changed_entity_mse"]
            <= (
                1
                - g[
                    "domain_only_development_changed_mse_reduction_from_initialization_each_target_each_seed"
                ]
            )
            * f["initial_training"]["changed_entity_mse"]
            for f in fits
            if f["identity"]["condition"] == "domain_only"
        ),
        "finite": all(f["composition"]["finite"] and f["gradient_audit_pass"] for f in fits),
    }
    return {
        "decision": "GO_TO_CONFIRMATION" if all(checks.values()) else "NO_GO",
        "gates": checks,
        "relative_changed_mse_gains": gains,
        "independent_composition_tasks_per_domain": 9,
        "optimizer_seeds": seeds,
    }
