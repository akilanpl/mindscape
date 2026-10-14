"""Replay scoring of externally sealed predictions, with task-level uncertainty."""
from __future__ import annotations

import math
import random
from collections import defaultdict

from .benchmark import Task, digest


def wilson(good: int, total: int) -> tuple[float, float]:
    if not 0 <= good <= total or total == 0:
        raise ValueError("invalid independent denominator")
    z = 1.959963984540054
    p = good / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - half, center + half


def paired_interval(task_gains: list[float], draws: int = 10000,
                    seed: int = 540052) -> tuple[float, float]:
    """Resample independent tasks after averaging matched seeds within each task."""
    if not task_gains or any(not math.isfinite(v) for v in task_gains) or draws < 100:
        raise ValueError("nonempty task gains and at least 100 draws required")
    rng = random.Random(seed)
    n = len(task_gains)
    samples = sorted(sum(rng.choice(task_gains) for _ in range(n)) / n for _ in range(draws))
    return samples[int(.025 * draws)], samples[min(draws - 1, int(.975 * draws))]


def replay(tasks: tuple[Task, ...], predictions: list[dict], labels: dict[str, list[int]],
           seal: dict, conditions: tuple[str, ...], seeds: tuple[int, ...]) -> dict:
    """Caller supplies released labels; this module never locates a lockbox.

    A hash verifies immutable bytes, not temporal precedence. Independent custodian
    must attest seal-before-label-release and approvals; absent attestation blocks.
    """
    if not tasks:
        raise ValueError("empty evaluation task list")
    if seal.get("prediction_hash") != digest(predictions):
        raise ValueError("prediction seal mismatch")
    if seal.get("label_commitment") != digest(labels):
        raise ValueError("label commitment mismatch")
    if seal.get("custodian_attested_precedence") is not True:
        raise ValueError("independent seal-before-release attestation required")
    if not conditions or not seeds or len(set(conditions)) != len(conditions) or len(set(seeds)) != len(seeds):
        raise ValueError("unique nonempty conditions and seeds required")
    task_map = {t.identity: t for t in tasks}
    if len(task_map) != len(tasks) or set(labels) != set(task_map):
        raise ValueError("identity/label coverage mismatch")
    for task in tasks:
        values = labels[task.identity]
        if len(values) != len(task.steps) or any(type(v) is not int for v in values):
            raise ValueError("invalid released reference")
    records = {}
    for row in predictions:
        if set(row) != {"identity", "condition", "seed", "states"}:
            raise ValueError("prediction record schema mismatch")
        key = (row["condition"], row["seed"], row["identity"])
        if key in records or row["identity"] not in task_map:
            raise ValueError("duplicate or unknown prediction")
        if row["condition"] not in conditions or row["seed"] not in seeds:
            raise ValueError("unexpected condition/seed")
        if len(row["states"]) != len(task_map[row["identity"]].steps) or any(
                type(v) is not int for v in row["states"]):
            raise ValueError("invalid prediction trace")
        records[key] = row["states"]
    if len(records) != len(tasks) * len(conditions) * len(seeds):
        raise ValueError("incomplete campaign; never drop failed seeds")
    groups = defaultdict(list)
    for task in tasks:
        groups[task.family].append(task)
    groups["overall"] = list(tasks)
    metrics = {}
    failure_traces = []
    for condition in conditions:
        metrics[condition] = {}
        for seed in seeds:
            metrics[condition][str(seed)] = {}
            for family, members in groups.items():
                exact = final_exact = intermediate = denominator = 0
                by_length = defaultdict(lambda: [0, 0])
                for task in members:
                    observed = records[condition, seed, task.identity]
                    target = labels[task.identity]
                    correct = observed == target
                    exact += correct
                    final_exact += observed[-1] == target[-1]
                    intermediate += sum(a == b for a, b in zip(observed, target))
                    denominator += len(target)
                    by_length[len(target)][0] += correct
                    by_length[len(target)][1] += 1
                    if family != "overall" and not correct:
                        failure_traces.append({"condition": condition, "seed": seed,
                                               "identity": task.identity, "family": family,
                                               "states": observed, "reference": target,
                                               "first_failure": next(i for i, (a, b) in
                                                                     enumerate(zip(observed, target)) if a != b)})
                metrics[condition][str(seed)][family] = {
                    "exact_correct": exact, "independent_tasks": len(members),
                    "exact": exact / len(members), "full_trace_exact": exact / len(members),
                    "final_answer_exact": final_exact / len(members), "wilson95": wilson(exact, len(members)),
                    "intermediate_correct": intermediate, "intermediate_denominator": denominator,
                    "intermediate_accuracy": intermediate / denominator,
                    "by_length": dict(by_length)}
    comparisons = {}
    if "recurrent" in conditions:
        for baseline in conditions:
            if baseline in ("recurrent", "procedural"):
                continue
            for family, members in groups.items():
                gains = [sum((records["recurrent", s, t.identity] == labels[t.identity]) -
                             (records[baseline, s, t.identity] == labels[t.identity])
                             for s in seeds) / len(seeds) for t in members]
                comparisons[f"{baseline}:{family}"] = {"mean_gain": sum(gains) / len(gains),
                    "paired_task_bootstrap95": paired_interval(gains), "independent_tasks": len(gains)}
    return {"metrics": metrics, "comparisons": comparisons, "failure_traces": failure_traces,
            "scientific_status": "UNVERIFIED", "prediction_hash": digest(predictions),
            "label_hash": digest(labels), "unique_tasks": len(tasks)}
