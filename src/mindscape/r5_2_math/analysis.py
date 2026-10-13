"""Post-hoc metrics and write-only trace replay; no checkpoint neural forwards."""

import math
import statistics

from .arithmetic import run
from .verification import verify


def wilson(correct, total):
    if not total:
        return None
    z = 1.96
    p = correct / total
    den = 1 + z * z / total
    center = (p + z * z / (2 * total)) / den
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return [center - radius, center + radius]


def analyze(records, protocol):
    metrics = {}
    for condition in (
        "recurrent",
        "reset_hidden",
        "memory_only",
        "zero_weight",
        "zero_weight_memory",
    ):
        metrics[condition] = {}
        for seed in protocol["optimizer_seeds"]:
            by = {}
            for family in ["development", *protocol["families"]]:
                rr = [
                    r
                    for r in records
                    if r["condition"] == condition
                    and r["seed"] == seed
                    and (
                        r["split"] == "development"
                        if family == "development"
                        else r["family"] == family
                    )
                ]
                good = sum(r["verification"]["exact"] for r in rr)
                local = sum(r["verification"]["local_correct"] for r in rr)
                total = sum(r["verification"]["local_total"] for r in rr)
                by[family] = {
                    "tasks": len(rr),
                    "correct": good,
                    "exact_accuracy": good / len(rr),
                    "exact_wilson95": wilson(good, len(rr)),
                    "local_correct": local,
                    "local_total": total,
                    "local_accuracy": local / max(1, total),
                    "strict_rollout_correct": sum(r["verification"]["strict"] for r in rr),
                    "sign_correct": sum(r["verification"]["sign_correct"] for r in rr),
                    "sign_total": sum(r["verification"]["sign_total"] for r in rr),
                    "stage_correct": sum(r["verification"]["stage_correct"] for r in rr),
                    "stage_total": sum(r["verification"]["stage_total"] for r in rr),
                    "local_seen_correct": sum(
                        d["correct"]
                        for r in rr
                        for d in r["verification"].get("context_diagnostics", [])
                        if d["memory_seen"]
                    ),
                    "local_seen_total": sum(
                        d["memory_seen"]
                        for r in rr
                        for d in r["verification"].get("context_diagnostics", [])
                    ),
                    "local_unseen_correct": sum(
                        d["correct"]
                        for r in rr
                        for d in r["verification"].get("context_diagnostics", [])
                        if not d["memory_seen"]
                    ),
                    "local_unseen_total": sum(
                        not d["memory_seen"]
                        for r in rr
                        for d in r["verification"].get("context_diagnostics", [])
                    ),
                }
            oo = [
                r
                for r in records
                if r["condition"] == condition and r["seed"] == seed and r["split"] == "ood"
            ]
            by["ood"] = {
                "tasks": len(oo),
                "correct": sum(r["verification"]["exact"] for r in oo),
                "exact_accuracy": sum(r["verification"]["exact"] for r in oo) / len(oo),
                "local_accuracy": sum(r["verification"]["local_correct"] for r in oo)
                / sum(r["verification"]["local_total"] for r in oo),
            }
            metrics[condition][str(seed)] = by
    mean = lambda c, f: statistics.mean(
        metrics[c][str(s)][f]["exact_accuracy"] for s in protocol["optimizer_seeds"]
    )
    g = protocol["gate"]
    checks = {
        "overall": mean("recurrent", "ood") >= g["minimum_mean_ood_exact"],
        "length": mean("recurrent", "length") >= g["minimum_length_exact"],
        "carry_borrow": mean("recurrent", "carry_borrow") >= g["minimum_carry_borrow_exact"],
        "sign": mean("recurrent", "sign") >= g["minimum_sign_exact"],
        "unseen_context": mean("recurrent", "unseen_local_context")
        >= g["minimum_unseen_context_exact"],
        "intermediate": statistics.mean(
            metrics["recurrent"][str(s)]["ood"]["local_accuracy"]
            for s in protocol["optimizer_seeds"]
        )
        >= g["minimum_intermediate_next_state_accuracy"],
        "reset_gain": mean("recurrent", "ood") - mean("reset_hidden", "ood")
        >= g["minimum_ood_gain_over_reset"],
        "memory_controls": all(
            mean("recurrent", "ood") - mean(c, "ood")
            >= g["minimum_ood_gain_over_memory_and_zero_weight_memory"]
            for c in ("memory_only", "zero_weight_memory")
        ),
        "seed_reset_gains": all(
            metrics["recurrent"][str(s)]["ood"]["exact_accuracy"]
            > metrics["reset_hidden"][str(s)]["ood"]["exact_accuracy"]
            for s in protocol["optimizer_seeds"]
        ),
    }
    return {
        "decision": "GO_TO_CONFIRMATION" if all(checks.values()) else "NO_GO",
        "gates": checks,
        "metrics": metrics,
        "independent_evaluation_tasks": protocol["development_tasks"]
        + len(protocol["families"]) * protocol["ood_tasks_per_family"],
        "independent_ood_tasks_per_family": protocol["ood_tasks_per_family"],
        "symbolic_tool_results_included_in_neural_metrics": False,
    }


def replay(manifest, records):
    tasks = {t["identity"]: t for t in manifest["tasks"]}
    for record in records:
        original = record["result"]
        events = iter(original["transitions"])
        stages = iter(original["stages"])

        class Saved:
            def reset(self):
                pass

            def local(self, state, saved_events=events):
                e = next(saved_events)
                if e["state"] != state:
                    raise ValueError("Local state trace mismatch")
                return e["written_digit"], e["written_carry"], e["prediction"]

            def sign(self, a, b, op, saved_stages=stages):
                stage = next(saved_stages)
                return stage["sign"], stage["sign_prediction"]

        result = run(tasks[record["identity"]], Saved())
        if (
            result != original
            or verify(tasks[record["identity"]], result) != record["verification"]
        ):
            raise ValueError("Replay outcome changed")
        if next(events, None) is not None or next(stages, None) is not None:
            raise ValueError("Extra recorded claim")
    return {
        "status": "PASS",
        "recorded_executions": len(records),
        "independent_tasks": len({r["identity"] for r in records}),
        "neural_re_forward": False,
    }
