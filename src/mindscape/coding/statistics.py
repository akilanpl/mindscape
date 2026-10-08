"""Task-paired uncertainty and explicitly censored learning-curve quantities."""

import math
from itertools import pairwise

import numpy as np


def bootstrap_mean(values, seed=17, repetitions=2000):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    rng = np.random.default_rng(seed)
    estimates = values[rng.integers(0, len(values), (repetitions, len(values)))].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": np.quantile(estimates, [0.025, 0.975]).tolist(),
        "n": len(values),
    }


def paired_delta(first, second):
    shared = sorted(set(first) & set(second))
    if set(first) != set(second):
        raise ValueError("Paired comparisons require identical task keys")
    return bootstrap_mean([second[k] - first[k] for k in shared])


def threshold(curve, target):
    """Minimum observed budget; no interpolation or extrapolation."""
    for budget, score in sorted(curve.items()):
        if score >= target:
            return {"budget": budget, "right_censored": False}
    return {"budget": None, "right_censored": True, "greater_than": max(curve)}


def der(baseline, system):
    if baseline["right_censored"] or system["right_censored"]:
        return {"value": None, "reason": "Threshold not reached in observed budgets"}
    if system["budget"] == 0:
        return {"value": None, "reason": "Zero denominator; 0/0 is undefined"}
    return {"value": baseline["budget"] / system["budget"], "reason": None}


def log_aulc(curve):
    pairs = sorted((math.log(n), s) for n, s in curve.items() if n > 0)
    if len(pairs) < 2:
        return None
    area = sum((x2 - x1) * (y1 + y2) / 2 for (x1, y1), (x2, y2) in pairwise(pairs))
    return area / (pairs[-1][0] - pairs[0][0])
