"""Post-execution changed-entity metrics; never part of the model input."""

import math


def summarize(target, prediction):
    delta = target["delta"]
    if len(delta) != len(prediction) or any(len(a) != len(b) for a, b in zip(delta, prediction)):
        raise ValueError("Effect dimensions")
    whole = []
    changed = []
    unchanged = []
    for actual, proposed in zip(delta, prediction, strict=True):
        if not all(math.isfinite(x) for x in actual + proposed):
            raise ValueError("Nonfinite effect")
        errors = [(x - y) ** 2 for x, y in zip(actual, proposed, strict=True)]
        whole.extend(errors)
        # Select whole entities (all output coordinates), not only individually nonzero coordinates.
        if any(abs(v) > 1e-8 for v in actual):
            changed.append(sum(errors) / len(errors))
        else:
            unchanged.append(sum(errors) / len(errors))
    return {
        "whole_graph_mse": sum(whole) / len(whole),
        "changed_entity_mse": sum(changed) / len(changed) if changed else None,
        "unchanged_entity_mse": sum(unchanged) / len(unchanged) if unchanged else None,
        "changed_entities": len(changed),
        "unchanged_entities": len(unchanged),
        "valid": target["valid"],
    }


def aggregate(records):
    def average(key):
        values = [r[key] for r in records if r[key] is not None]
        return sum(values) / len(values) if values else None

    return {
        "tasks": len(records),
        "tasks_with_changed_entities": sum(r["changed_entities"] > 0 for r in records),
        "positive_valid": sum(r["valid"] for r in records),
        "negative_invalid": sum(1 - r["valid"] for r in records),
        "whole_graph_mse": average("whole_graph_mse"),
        "changed_entity_mse": average("changed_entity_mse"),
        "unchanged_entity_mse": average("unchanged_entity_mse"),
    }


def baselines(task, target):
    d = task["domain"]
    p = task["public"]
    persistence = [[0.0] * len(row) for row in target["delta"]]
    local = [row.copy() for row in persistence]
    # Simple baseline uses public action/state only; independently infer bounds, never target.valid.
    if d == "mathematics" and 0 <= p["destination"] < len(p["values"]):
        sign = 1 if p["operation"] == "add" else -1
        local[p["destination"]][0] = (
            sign * p["amount"] / 100
        )  # unit-source heuristic, ignores operand value
    elif d == "coding" and 0 <= p["target_leaf"] < len(p["constants"]):
        local[p["target_leaf"] + 1][0] = (
            p["increment"] / 100
        )  # direct literal effect only; no AST propagation
    elif d == "physics" and 0 <= p["destination"] < len(p["masses"]):
        i = p["destination"]
        local[i][2:] = [
            v / 4 for v in p["impulse"]
        ]  # unit-mass impulse; ignores real mass/position evolution
    return {"persistence": summarize(target, persistence), "local_action": summarize(target, local)}
