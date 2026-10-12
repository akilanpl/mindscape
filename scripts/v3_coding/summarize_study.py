"""Complete-coverage descriptive results, task-paired uncertainty and resource totals."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from mindscape.v3.coding_learning.agent import CONDITIONS
from mindscape.v3.coding_learning.analysis import analyze


def summarize(root):
    root = Path(root)
    summaries, traces = {}, {}
    for seed in (11, 23, 37):
        run = root / f"transport_v2_s{seed}"
        summary = analyze(run, root / f"analysis-v2-s{seed}.json")
        if summary["episodes"] != 144:
            raise ValueError(f"Incomplete seed{seed}: {summary['episodes']}/144")
        summaries[seed] = summary
        traces[seed] = [
            json.loads(p.read_text())["trace"]
            for p in run.glob("*.json")
            if len(p.name.split("-")[0]) == 64
        ]
    cells = [
        dict(seed=seed, **cell) for seed, summary in summaries.items() for cell in summary["cells"]
    ]
    fields = (
        "seed",
        "condition",
        "split",
        "successes",
        "total",
        "accuracy",
        "first_edit_successes",
        "self_corrected",
        "false_visible_repairs",
        "parse_failures",
        "model_calls",
        "visible_vm_calls",
        "private_vm_calls",
        "tokens_in_new",
        "tokens_out_new",
        "visible_brier",
    )
    with (root / "ablation-results.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(cells)
    by_key = {
        (seed, r["task_id"], r["condition"]): r for seed, rows in traces.items() for r in rows
    }
    ood_ids = sorted({r["task_id"] for r in traces[11] if r["split"] == "ood"})
    rng = np.random.default_rng(840014)
    across_seed = []
    for baseline in CONDITIONS:
        if baseline == "mindscape":
            continue
        differences = np.asarray(
            [
                np.mean(
                    [
                        int(by_key[seed, task, "mindscape"]["success"])
                        - int(by_key[seed, task, baseline]["success"])
                        for seed in (11, 23, 37)
                    ]
                )
                for task in ood_ids
            ]
        )
        samples = differences[rng.integers(0, len(differences), (2000, len(differences)))].mean(
            axis=1
        )
        across_seed.append(
            {
                "baseline": baseline,
                "task_clusters": len(ood_ids),
                "mean_seed_delta": float(differences.mean()),
                "paired_task_bootstrap95": list(np.quantile(samples, [0.025, 0.975])),
            }
        )
    families = defaultdict(lambda: {"successes": 0, "total": 0})
    candidate_counts = Counter()
    changed_first = Counter()
    failures = []
    for seed, rows in traces.items():
        for row in rows:
            key = (seed, row["condition"], row["family"])
            families[key]["successes"] += int(row["success"])
            families[key]["total"] += 1
            for attempt in row["attempts"]:
                candidate_counts[str(len(attempt["candidates"]))] += 1
                if attempt["selected"] not in (None, 0):
                    changed_first[row["condition"]] += 1
            if not row["success"]:
                failures.append(
                    {
                        "seed": seed,
                        "task_id": row["task_id"],
                        "condition": row["condition"],
                        "family": row["family"],
                        "split": row["split"],
                        "errors": [a["parse_error"] for a in row["attempts"] if a["parse_error"]],
                        "private_passed": row["terminal_private"]["passed"],
                        "private_total": row["terminal_private"]["total"],
                    }
                )
    invocations = [
        dict(seed=seed, **json.loads(p.read_text()))
        for seed in (11, 23, 37)
        for p in (root / f"transport_v2_s{seed}").glob("invocation-*.json")
    ]
    resources = {
        "invocations": len(invocations),
        "fresh_requests": sum(r["model_calls"] for r in invocations),
        "fresh_input_tokens": sum(r["tokens_in"] for r in invocations),
        "fresh_output_tokens": sum(r["tokens_out"] for r in invocations),
        "summed_invocation_wall_seconds": sum(r["elapsed_seconds"] for r in invocations),
        "exclusive_generation_batch_seconds": sum(
            b["wall_seconds"] for r in invocations for b in r["batches"]
        ),
        "peak_rss_bytes": max(r["peak_rss_bytes"] for r in invocations),
    }
    primary = {c["condition"]: c for c in summaries[11]["cells"] if c["split"] == "ood"}
    best = max(c["accuracy"] for k, c in primary.items() if k != "mindscape")
    strongest = [k for k, c in primary.items() if k != "mindscape" and c["accuracy"] == best]
    primary_pairs = [
        p for p in summaries[11]["paired"] if p["split"] == "ood" and p["baseline"] in strongest
    ]
    target = primary["mindscape"]["accuracy"] >= 0.6
    superiority = all(p["delta"] >= 0.1 and p["paired_bootstrap95"][0] > 0 for p in primary_pairs)
    result = {
        "episodes": 432,
        "unique_evaluation_tasks": 18,
        "unique_ood_tasks": 12,
        "seeds": [11, 23, 37],
        "cells": cells,
        "resources": resources,
        "primary": {
            "mindscape": primary["mindscape"],
            "strongest_conditions": strongest,
            "strongest_accuracy": best,
            "pairs": primary_pairs,
            "target60_met": target,
            "paired_superiority_gate_met": superiority,
            "joint_research_gate_met": target and superiority,
        },
        "seed_averaged_task_paired": across_seed,
        "families": [
            dict(seed=k[0], condition=k[1], family=k[2], **v) for k, v in sorted(families.items())
        ],
        "candidate_set_sizes_all_attempts": dict(candidate_counts),
        "selections_other_than_candidate_zero": dict(changed_first),
        "failure_rows": failures,
        "limits": "Fixed18-task synthetic study. Critic seeds share greedy Qwen cache; not432 independent tasks or three independent LM samples. Unadjusted descriptive intervals.",
    }
    (root / "study-summary.json").write_text(json.dumps(result, indent=2))
    table = [
        "| Condition | IID s11 | OOD s11 | OOD s23 | OOD s37 | First edit OOD s11 | False visible OOD s11 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:

        def cell(seed, split, selected_condition=condition):
            return next(
                c
                for c in summaries[seed]["cells"]
                if c["condition"] == selected_condition and c["split"] == split
            )

        iid, ood = cell(11, "iid"), cell(11, "ood")
        table.append(
            f"| {condition} | {iid['successes']}/6 | {ood['successes']}/12 | "
            f"{cell(23, 'ood')['successes']}/12 | {cell(37, 'ood')['successes']}/12 | "
            f"{ood['first_edit_successes']}/12 | {ood['false_visible_repairs']} |"
        )
    (root / "ablation-table.md").write_text("\n".join(table) + "\n")
    print(json.dumps({"primary": result["primary"], "resources": resources}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results/v3_coding")
    summarize(p.parse_args().root)
