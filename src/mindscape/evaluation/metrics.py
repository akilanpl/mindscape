"""Independent metric utilities. Empty denominators are undefined."""
import math


def rate(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def aggregate(records):
    records = list(records)
    grounded = rate(r.grounded for r in records)
    return {"accuracy": rate(r.correct for r in records),
            "trajectory_validity": rate(r.trajectory_valid for r in records
                                        if r.predicted_trajectory is not None),
            "goal_success_rate": rate(r.goal_reached for r in records),
            "grounded_rate": grounded,
            "unsupported_rate": None if grounded is None else 1 - grounded,
            "example_count": len(records),
            "trajectory_count": sum(r.predicted_trajectory is not None for r in records)}


def n_star(curve, alpha):
    if not 0 <= alpha <= 1:
        raise ValueError("Accuracy target must be in [0,1]")
    for n, accuracy in curve.items():
        if type(n) is not int or n <= 0 or not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
            raise ValueError("Invalid observed learning curve")
    reached = [n for n, accuracy in curve.items() if accuracy >= alpha]
    return min(reached) if reached else "not reached"


def data_efficiency_ratio(baseline_curve, mindscape_curve, alpha):
    baseline, mindscape = n_star(baseline_curve, alpha), n_star(mindscape_curve, alpha)
    return "not reached" if "not reached" in (baseline, mindscape) else baseline / mindscape
