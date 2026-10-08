"""Populate final claims and report exclusively from completed measured artifacts."""

import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from mindscape.coding.statistics import paired_delta

base = Path("results/coding")
for name in (
    "completion_lockbox_eval_v1",
    "completion_trace_audit_v1",
    "latency_probe_v1",
    "stress_v1",
    "completion_gradient_v1",
):
    if not (base / name / "complete.json").exists():
        raise RuntimeError("Incomplete " + name)
if not Path("results/sql/portability_v1/complete.json").exists():
    raise RuntimeError("SQL study incomplete")
metrics = json.loads((base / "completion_analysis_v1/metrics.json").read_text())
fairness = json.loads((base / "completion_audits_v1/fairness_leakage.json").read_text())
test = json.loads(Path("experiments/coding_completion_v2/test_results.json").read_text())
if test["failures"] or test["errors"]:
    raise RuntimeError("Final tests failed")
read = lambda folder: [json.loads(s) for s in (folder / "rows.jsonl").read_text().splitlines()]
final = read(base / "completion_lockbox_eval_v1")
trace = read(base / "completion_trace_audit_v1")
stress = read(base / "stress_v1")
sql = read(Path("results/sql/portability_v1"))
public = {
    size: json.loads((base / f"humaneval_{size}_v1/summary.json").read_text())
    for size in ("05b", "15b")
}
training = []
for path in sorted(
    list((base / "gradient_v1").glob("seed_*/training.json"))
    + list((base / "completion_gradient_v1").glob("*/seed_*/training.json"))
):
    record = json.loads(path.read_text())
    training.append({"path": str(path), **record})
comparisons = {}
for split in ("test", "ood_test"):
    first = {
        r["task_id"]: int(r["success"])
        for r in final
        if r["condition"] == "model_only" and r["split"] == split
    }
    second = {
        r["task_id"]: int(r["success"])
        for r in final
        if r["condition"] == "mindscape_c" and r["split"] == split
    }
    comparisons[split] = paired_delta(first, second)
ablations = {}
full = {
    r["task_id"]: int(r["success"])
    for r in stress
    if r["kind"] == "ablation" and r["variant"] is None
}
for variant in ("no_memory", "no_dream"):
    removed = {
        r["task_id"]: int(r["success"])
        for r in stress
        if r["kind"] == "ablation" and r["variant"] == variant
    }
    # Positive effect means full exceeds removed.
    ablations[variant] = paired_delta(removed, full)
c = metrics["final"]["mindscape_c"]
sql_summary = {
    condition: {
        "tasks": len(rows),
        "successes": sum(r["success"] for r in rows),
        "accuracy": float(np.mean([r["success"] for r in rows])),
    }
    for condition, rows in (
        (name, [r for r in sql if r["condition"] == name]) for name in ("model_only", "mindscape_c")
    )
}
ratios = [
    q["value"]
    for key, values in metrics["DER"].items()
    if key.startswith("mindscape_c|")
    for q in values.values()
    if q["value"] is not None
]
claims = [
    ("Bounded vertical reaches 100%", c["successes"] == 100, f"{c['successes']}/100 locked tasks"),
    ("Mindscape improves IID", comparisons["test"]["ci95"][0] > 0, str(comparisons["test"])),
    (
        "Mindscape improves OOD",
        comparisons["ood_test"]["ci95"][0] > 0,
        str(comparisons["ood_test"]),
    ),
    (
        "Mindscape achieves at least 5x data efficiency",
        len(ratios) == 6 and min(ratios) >= 5,
        "Observed DER only; censored/zero thresholds do not establish this claim",
    ),
    (
        "Mindscape grounded success reaches 99%",
        c["grounded_success"]["mean"] >= 0.99,
        f"Conservative goal-and-legal-episode rate {c['grounded_success']['mean']:.6f}",
    ),
    (
        "Memory contributes measurable success",
        ablations["no_memory"]["ci95"][0] > 0,
        str(ablations["no_memory"]),
    ),
    (
        "Dream simulation contributes measurable success",
        ablations["no_dream"]["ci95"][0] > 0,
        str(ablations["no_dream"]),
    ),
    (
        "Mindscape transfers vertically with at least 95% SQL success",
        sql_summary["mindscape_c"]["accuracy"] >= 0.95,
        str(sql_summary["mindscape_c"])
        + "; interface demonstration, not matched causal learned transfer",
    ),
    (
        "Mindscape exceeds published GPT4 HumanEval reference",
        public["15b"]["pass@1"] > 0.67,
        "Local public model score; prompt/runtime differ, no broad superiority inference",
    ),
    (
        "Mindscape exceeds published Gemini Ultra HumanEval reference",
        public["15b"]["pass@1"] > 0.744,
        "Local public model score; prompt/runtime differ, no broad superiority inference",
    ),
]
claim_records = [
    {"claim": name, "result": "supported" if supported else "not supported", "evidence": evidence}
    for name, supported, evidence in claims
]
failures = Counter()
for row in final:
    if row["success"]:
        continue
    if any(a.get("parse_error") for a in row.get("attempts", [])):
        category = "invalid_model_action"
    elif row["terminal"]["execution"]["timeout"]:
        category = "resource_timeout"
    elif row["terminal"]["execution"]["returncode"]:
        category = "execution_error"
    else:
        category = "hidden_test_mismatch"
    failures[(row["condition"], category)] += 1
scorecard = {
    "release_tag": "mindscape-research-v2",
    "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "capability": metrics["final"],
    "generalization": {"paired_final_deltas": comparisons, "stress": metrics["stress"]},
    "data_efficiency": {
        k: metrics[k]
        for k in ("thresholds", "DER", "normalized_log_AULC", "learning_cells", "paired_deltas")
    },
    "grounding": {
        "independent_replayed_episodes": len(trace),
        "replay_goal_agreement": all(r["matches_reported_goal"] for r in trace),
        "independent_trajectory_valid_rate": float(np.mean([r["trajectory_valid"] for r in trace])),
        "unsupported_structured_result_rate": float(
            np.mean([not r["matches_reported_goal"] for r in trace])
        ),
        "free_text_claims": "Not scored; demo omits raw prose",
    },
    "reliability": {
        "recovery": metrics["recovery"],
        "failure_categories": {
            condition + "|" + category: count for (condition, category), count in failures.items()
        },
    },
    "latency": {
        "fresh": metrics["latency_fresh"],
        "median_overhead": metrics["fresh_median_overhead"],
    },
    "compute": {
        "training_records": training,
        "total_measured_training_seconds": sum(r["training_seconds"] for r in training),
        "peak_training_rss_bytes": max(r["peak_rss_bytes"] for r in training),
        "gpu_used": False,
        "vram_allocated_bytes": 0,
        "paid_services_usd": 0,
        "training_time_scope": "Training loop/checkpoint writes; model loading excluded",
        "larger_gradient_budgets": "250/500/1000/2500/5000 not executed; CPU RAM/time scope, no extrapolated results",
    },
    "external": {
        "measured_humaneval": public,
        "historical_humaneval_percent": {
            "GPT3.5": 48.1,
            "GPT4": 67.0,
            "Gemini1.0Pro": 67.7,
            "Gemini1.0Ultra": 74.4,
        },
        "historical_protocol_comparability": "Incomplete; see sourced report",
    },
    "second_vertical": sql_summary,
    "ablations": ablations,
    "reproducibility": {
        "tests": test,
        "fairness_checks": fairness["check_count"],
        "local_leakage": fairness["local_leakage_status"],
        "pretraining_contamination": "UNKNOWN",
        "overall_contamination_free": "FAIL / not established",
        "main_gradient_seeds": [11, 23, 37],
        "additional_patch_seeds": [53, 71],
    },
    "claims": claim_records,
    "limitations": metrics["limitations"],
}
root = Path("experiments/coding_completion_v2")
scorecard["targets"] = {
    "bounded_100_percent": c["successes"] == 100,
    "IID_at_least_95_percent": c["split_accuracy"]["test"] >= 0.95,
    "OOD_at_least_90_percent": c["split_accuracy"]["ood_test"] >= 0.90,
    "grounded_at_least_99_percent": c["grounded_success"]["mean"] >= 0.99,
    "HumanEval_at_least_75_percent": public["15b"]["pass@1"] >= 0.75,
    "TTFT_p50_under_2_seconds": metrics["latency_fresh"]["mindscape_c"]["first_call_ttft"][
        "p50_p95_p99"
    ][0]
    < 2,
    "TTFT_p95_under_5_seconds": metrics["latency_fresh"]["mindscape_c"]["first_call_ttft"][
        "p50_p95_p99"
    ][1]
    < 5,
    "repair_p50_under_10_seconds": metrics["latency_fresh"]["mindscape_c"]["wall_seconds"][
        "p50_p95_p99"
    ][0]
    < 10,
    "repair_p95_under_30_seconds": metrics["latency_fresh"]["mindscape_c"]["wall_seconds"][
        "p50_p95_p99"
    ][1]
    < 30,
    "median_overhead_under_2": metrics["fresh_median_overhead"]["mindscape_c"] < 2,
}
(root / "scorecard.json").write_text(json.dumps(scorecard, indent=2))
(root / "claims.json").write_text(json.dumps(claim_records, indent=2))
docs = Path("docs/coding")
claim_text = (
    "# Final claim table\n\nClaims refer to these bounded experimental protocols; observed improvements do not isolate architecture from supervision/feedback/compute.\n\n| Claim | Evidence | Result |\n|---|---|---|\n"
    + "".join(
        f"| {r['claim']} | {r['evidence'].replace('|', '/')} | {r['result']} |\n"
        for r in claim_records
    )
)
(docs / "final_claims.md").write_text(claim_text)
lines = [
    "# Final coding research results",
    "",
    f"The frozen Mindscape C configuration passed **{c['successes']}/100 tasks ({c['success']['mean'] * 100:.2f}%)**. The 100% target is {'met on this finite lockbox' if c['successes'] == 100 else 'not met'}. This does not establish universal or general software-engineering perfection.",
    "",
    "| Condition | Success /100 | IID | Structural OOD | Per-test pass | Grounded success |",
    "|---|---:|---:|---:|---:|---:|",
]
for condition, value in metrics["final"].items():
    lines.append(
        f"| {condition} | {value['successes']} | {value['split_accuracy']['test'] * 100:.2f}% | {value['split_accuracy']['ood_test'] * 100:.2f}% | {value['per_test_pass_rate'] * 100:.2f}% | {value['grounded_success']['mean'] * 100:.2f}% |"
    )
lines += [
    "",
    "The final comparisons are paired over identical independent tasks. Each final model uses the preregistered100-task training budget/seed11. Three-seed learning curves use a separate diagnostic pool; confidence intervals and all measured cells are in the scorecard. Causal architecture isolation is not established.",
    "",
    f"HumanEval:0.5B {public['05b']['successes']}/164 ({public['05b']['pass@1'] * 100:.2f}%);1.5B {public['15b']['successes']}/164 ({public['15b']['pass@1'] * 100:.2f}%). Both use local greedy one-completion512-token instruction prompts and WASI; no feedback or public-score tuning.",
    "",
    "## Data efficiency",
    "",
    json.dumps({k: metrics[k] for k in ("thresholds", "DER", "normalized_log_AULC")}, indent=2),
    "",
    "Unreached thresholds are censored. Undefined DER is not1x or infinity. Collecting5,000 verified trajectories does not mean training5,000-example gradient models. Larger gradient budgets were not executed on this CPU-only16GB system; measured training times and peak memory are recorded below.",
    "",
    f"Measured training-loop time across retained main adapters: {scorecard['compute']['total_measured_training_seconds']:.3f}s; highest recorded peak RSS: {scorecard['compute']['peak_training_rss_bytes']} bytes. These are measurements, not projections of larger runs.",
    "",
    "## Recovery and ablations",
    "",
    json.dumps({"recovery": metrics["recovery"], "stress": metrics["stress"]}, indent=2),
    "",
    "## Fresh latency and compute",
    "",
    json.dumps(scorecard["latency"], indent=2),
    "",
    "Nested stage timers are inclusive and cannot be added as exclusive costs. Fresh latency contains no generation cache hits. P99 from14 tasks/condition is exploratory. Ablations preserve weights except the explicitly labeled patch-only-adapter diagnostic. Removing terminal private assessment would remove the measurement; the goal-text ablation is not that intervention.",
    "",
    f"SQL portability: {json.dumps(sql_summary)}. Thirty instances from five simple query templates, zero SQL training. This measures a small interface demonstration, not general SQL competence.",
    "",
    "## Audits and limitations",
    "",
    f"Local leakage checks: {fairness['local_leakage_status']}, {fairness['check_count']} checks. Independent trajectory replays:{len(trace)}. Overall contamination-free claim: FAIL/not established because foundation pretraining exposure is unknown.",
    "",
    f"Tests: {test['tests']} passed; failures:{test['failures']}, errors:{test['errors']}. Historical multiplication reports remain unchanged in their original sections and release.",
    "",
    "## Claims",
    "",
    claim_text,
    "",
    "## Repository and release",
    "",
    f"Source commit at report generation: `{scorecard['source_commit']}`. Final release commit is resolved with `git rev-parse mindscape-research-v2` after immutable snapshot verification; it is not inserted retroactively into hash-frozen evidence.",
    "",
    "```text",
    "src/mindscape/{core,data,evaluation,models,coding,sql}/",
    "scripts/coding/",
    "configs/coding/",
    "docs/coding/",
    "experiments/coding_completion_v2/",
    "demo/coding/",
    "results/final/coding_research_v2/",
    "tests/",
    "```",
    "",
    "Reproduction commands: [reproducibility](reproducibility.md). Historical references: [sourced comparison](historical_gpt_gemini_comparison.md). Machine scorecard: `experiments/coding_completion_v2/scorecard.json`.",
]
(docs / "final_results.md").write_text("\n".join(lines) + "\n")
(docs / "final_metrics.md").write_text(
    "# Final measured metrics\n\nAll values, counts, intervals, censored thresholds and protocol caveats are in [the scorecard](../../experiments/coding_completion_v2/scorecard.json) and [the final report](final_results.md). Fresh latency and recorded research timing are separate.\n"
)
(docs / "final_limitations.md").write_text(
    "# Final limitations\n\n"
    + "\n".join("- " + v for v in scorecard["limitations"])
    + "\n\nAdditional limits: small generated repositories, four hidden cases/task, few semantic templates, three adapter seeds, offline expert replay of repair actions only, static hypothetical planning, CPU-only training capped at100 unique tasks, tiny SQL portability study, noncanonical public runtime, and unknown foundation-model contamination. No general superiority over GPT/Gemini or architecture-only causal advantage is established.\n"
)
for name in (
    "final_architecture",
    "final_experimental_protocol",
    "final_results",
    "final_metrics",
    "final_limitations",
    "final_claims",
    "historical_gpt_gemini_comparison",
    "reproducibility",
):
    path = Path("docs") / (name + ".md")
    link = f"[Coding research completion](coding/{name}.md)"
    if path.exists():
        text = path.read_text()
        if link not in text:
            path.write_text(
                text
                + "\n\n## Coding research completion\n\n"
                + link
                + " preserves the historical study above and records the separate coding/SQL completion evidence.\n"
            )
    else:
        path.write_text("# " + name.replace("_", " ").title() + "\n\n" + link + "\n")
readme = Path("README.md")
text = readme.read_text()
link = "docs/coding/final_results.md"
if link not in text:
    readme.write_text(
        text
        + "\n\n## Coding research completion v2\n\n[Measured final report](docs/coding/final_results.md) · [Claims](docs/coding/final_claims.md) · [Reproduction: datasets, training, evaluation, final benchmark and demo](docs/coding/reproducibility.md) · [Machine scorecard](experiments/coding_completion_v2/scorecard.json). Historical multiplication evidence remains intact. The recorded demo is `demo/coding/index.html`; the immutable completion snapshot is `results/final/coding_research_v2`.\n"
    )
print("Wrote measured scorecard, claims, final report and documentation links")
