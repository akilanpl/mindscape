"""Build final R6 reports only from two verified fits and completed CPU evaluation.

This controller performs no inference or fitting and never chooses a model from
final scores. Existing minimum-development selections are used unchanged.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pinned(root, name, expected):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file() or sha(path) != expected:
        raise ValueError("Missing or hash-mismatched artifact: " + str(name))
    return path


def selected_development(receipt):
    step = receipt["best_development_step"]
    if step == 0:
        return receipt["initial_development"]
    matches = [row for row in receipt["dev_history"] if row["update"] == step]
    if len(matches) != 1:
        raise ValueError("Development selection has no unique recorded metrics")
    return matches[0]


def learning_gates(receipt):
    selected = selected_development(receipt)["native_autoregressive_metrics"]
    initial = receipt["initial_development"]["native_autoregressive_metrics"]
    base = receipt["development_training_only_baseline"]
    gates = {}
    for domain in ("mathematics", "coding"):
        observed = selected[domain]["exact_accuracy"]
        gates[domain] = {"passed": observed >= base[domain]["exact_accuracy"] + .05 and observed > initial[domain]["exact_accuracy"],
                         "selected_dev": observed, "initial_dev": initial[domain]["exact_accuracy"],
                         "training_prior_dev": base[domain]["exact_accuracy"], "metric": "exact_accuracy"}
    observed = selected["physics"]["mse"]
    gates["physics"] = {"passed": observed <= .8 * base["physics"]["mse"] and observed < initial["physics"]["mse"],
                        "selected_dev": observed, "initial_dev": initial["physics"]["mse"],
                        "training_prior_dev": base["physics"]["mse"], "metric": "physical-state MSE"}
    return gates


def _json_block(value):
    return "```json\n" + json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n```\n"


def _number(value):
    return "unavailable" if value is None else f"{value:.6g}"


def output_diversity(records):
    """Count actual emitted answers, not reference labels or decoder logits."""
    result = {}
    for domain in ("mathematics", "coding"):
        for regime in ("all-final", "final_iid", "final_ood"):
            rows = [row for row in records if row["domain"] == domain and (regime == "all-final" or row["split"] == regime)]
            counts = Counter(row["prediction"].get("answer") for row in rows)
            valid = {answer: count for answer, count in counts.items() if isinstance(answer, str)}
            ordered = sorted(valid.items(), key=lambda item: (-item[1], item[0]))
            result[regime + "/" + domain] = {
                "count": len(rows), "distinct_valid_answers": len(valid),
                "invalid_answers": sum(count for answer, count in counts.items() if not isinstance(answer, str)),
                "answer_counts": dict(ordered),
                "most_common_fraction": ordered[0][1] / len(rows) if ordered and rows else None,
                "exact_correct": sum(row["prediction"].get("answer") == row["target"]["answer"] for row in rows),
            }
    return result


def failure_interpretation(evaluation, evaluation_root):
    """Describe observed collapse and matched physics changes from pinned files."""
    selected = [entry for entry in evaluation["evaluations"] if entry["role"] in ("best", "best-dev")]
    diversity = {entry["fit_id"]: output_diversity(read(evaluation_root / entry["prediction_file"])) for entry in selected}
    lines = ["## Observed output collapse and physical-law failures", "",
             "Native sequential arithmetic and program semantics have not been demonstrated. "
             "The development-selected models repeatedly emit a few answer strings across different protected problems. "
             "A rare correct string under this collapse is not evidence of transferable arithmetic. "
             "The raw coding model's perfect rename/reformat invariance accompanies constant wrong predictions; it does not establish program understanding.", ""]
    for fit, groups in diversity.items():
        for domain in ("mathematics", "coding"):
            stats = groups["all-final/" + domain]
            lines.append(f"- {fit}, {domain}: {stats['distinct_valid_answers']} distinct valid outputs across {stats['count']} final cases; "
                         f"answer counts {json.dumps(stats['answer_counts'], sort_keys=True)}; {stats['exact_correct']} exact answers.")
    lines += ["", "These counts come from the verified raw prediction files and include both final regimes. "
              "Teacher-forced loss reduction did not produce reliable free-running reasoning.", ""]
    by_fit = {entry["fit_id"]: entry for entry in selected}
    raw, scaled = by_fit["r6-p1-raw"], by_fit["r6-p1-scaled"]
    lines += ["The physics correction changes conditioning, with the same architecture, seed, cases and update budget. "
              "Matched protected state errors improved, but intervention results remain mixed:", "",
              "| Matched physics measure | Raw selected | Standardized selected |",
              "|---|---:|---:|"]
    for split in ("final_iid", "final_ood"):
        first = raw["metrics"]["per_split"][split + "/physics"]["mse"]
        second = scaled["metrics"]["per_split"][split + "/physics"]["mse"]
        lines.append(f"| {split} state MSE | {_number(first)} | {_number(second)} |")
    for probe in ("final_iid/physics/double_time", "final_iid/physics/split_time_same_force"):
        first, second = raw["perturbation_summary"][probe], scaled["perturbation_summary"][probe]
        lines.append(f"| {probe} response MSE | {_number(first['response_mse'])} | {_number(second['response_mse'])} |")
    split_raw = raw["perturbation_summary"]["final_iid/physics/split_time_same_force"]
    split_scaled = scaled["perturbation_summary"]["final_iid/physics/split_time_same_force"]
    lines += ["", "Splitting a constant-force time interval leaves the analytical final state unchanged. "
              f"Neural predictions changed on {split_raw['prediction_changed_fraction']:.1%} of raw and {split_scaled['prediction_changed_fraction']:.1%} of corrected IID probes. "
              "The residual response errors and worsened double-time response error prevent a physical-law compositionality claim. "
              "Average state-error improvement is evidence of better fitted predictions, not recovered mechanics or general STEM reasoning.", ""]
    return "\n".join(lines), diversity


def collect(outputs):
    """Fail closed before writing any final report when receipts are incomplete."""
    protocol = read(outputs / "R6_PROTOCOL_LOCK.json")
    if protocol["total_optimizer_update_ceiling"] != 1200 or protocol["updates_per_fit"] != 600:
        raise ValueError("Unsupported/unlocked pilot budget")
    required_gate_text = {"mathematics": "5 percentage points", "coding": "5 percentage points", "physics": "20 percent"}
    if any(text not in protocol["gate_2_by_domain"][domain] for domain, text in required_gate_text.items()):
        raise ValueError("Gate thresholds differ from reviewed locked protocol")
    manifest = read(outputs / "dataset-v1/manifest.json")
    for name, metadata in manifest["files"].items():
        pinned(outputs / "dataset-v1", name, metadata["sha256"])
    fits, evidence = {}, {}
    for key in ("p1-raw", "p1-scaled"):
        folder = outputs / key / "fit"
        receipt = read(folder / "receipt.json")
        proof = read(outputs / key / "trainer-evidence.json")
        if (receipt["status"] != "COMPLETED" or receipt["optimizer_updates"] != 600
                or not receipt["checkpoint_reload_verified"] or not receipt["per_parameter_adam_steps_verified"]
                or receipt["hardware"]["device"] != "cuda:0"):
            raise ValueError("Actual completed verified CUDA fit required")
        if not proof["source_verified"] or not proof["evaluation_data_absent"] or proof["fit_receipt_sha256"] != sha(folder / "receipt.json"):
            raise ValueError("Trainer control-plane evidence mismatch")
        for name, digest in receipt["output_hashes"].items():
            pinned(folder, name, digest)
        for name, digest in (("initial.json", receipt["initial_sha256"]), ("final.json", receipt["checkpoint_sha256"]),
                             ("best-dev.json", receipt["selected_checkpoint_sha256"])):
            pinned(folder, name, digest)
        if receipt["training_sha256"] != manifest["files"]["train.json"]["sha256"] or receipt["development_sha256"] != manifest["files"]["dev.json"]["sha256"]:
            raise ValueError("Trainer data identities differ")
        selected_development(receipt)
        fits[key], evidence[key] = receipt, proof
    first, second = fits.values()
    if first["config"]["seed"] != second["config"]["seed"]:
        raise ValueError("Correction seed differs")
    config_exceptions = {"fit_id", "physics_target_mean", "physics_target_scale"}
    if {k:v for k,v in first["config"].items() if k not in config_exceptions} != {k:v for k,v in second["config"].items() if k not in config_exceptions}:
        raise ValueError("Correction changes beyond locked target normalization")
    for name in ("research/r6/common.py", "research/r6/architectures/model.py", "research/r6/training/worker.py"):
        original = evidence["p1-raw"]["payload_files_sha256"].get(name)
        if not isinstance(original, str) or not re.fullmatch(r"[0-9a-f]{64}", original) or original != evidence["p1-scaled"]["payload_files_sha256"].get(name):
            raise ValueError("Correction source differs: " + name)
    decision = read(outputs / "CORRECTION_DECISION_BEFORE_FINAL.json")
    if decision["first_fit_receipt_sha256"] != sha(outputs / "p1-raw/fit/receipt.json"):
        raise ValueError("Correction diagnosis is not linked to raw pilot")
    p0 = read(outputs / "p1-raw/p0-verification.json")
    if p0["status"] != "PASSED_EXISTING_CHECKPOINT_EQUIVALENCE" or p0["optimizer_updates"] != 0:
        raise ValueError("P0 checkpoint equivalence proof required")
    execution = outputs / "independent-evaluation"
    proof = read(execution / "controller-verification.json")
    evaluation_root = execution / "downloads/r6-evaluation-output"
    if not evaluation_root.exists():
        evaluation_root = execution / "r6-evaluation-output"
    evaluation = read(evaluation_root / "evaluation_receipt.json")
    if (evaluation["status"] != "COMPLETED" or evaluation["device"] != "cpu" or evaluation["optimizer_updates_executed"] != 0
            or proof["status"] != "VERIFIED" or not proof["control_plane_source_verified"]
            or proof["receipt_sha256"] != sha(evaluation_root / "evaluation_receipt.json")):
        raise ValueError("Completed externally verified zero-update CPU evaluation required")
    for name, digest in evaluation["artifacts"].items():
        pinned(evaluation_root, name, digest)
    if set(evaluation["dataset_sha256"].values()) != {manifest["files"]["protected/"+split+".json"]["sha256"] for split in ("final_iid", "final_ood")}:
        raise ValueError("Final evaluator dataset identities differ")
    expected = {(receipt["fit_id"], role): receipt["initial_sha256" if role == "initial" else "selected_checkpoint_sha256"]
                for receipt in fits.values() for role in ("initial", "best")}
    actual = {(entry["fit_id"], entry["role"]): entry["checkpoint_sha256"] for entry in evaluation["evaluations"]}
    # The controller may spell the frozen role best-dev instead of best.
    actual = {(fit, "best" if role == "best-dev" else role): digest for (fit, role), digest in actual.items()}
    if len(evaluation["evaluations"]) != 4 or actual != expected:
        raise ValueError("Evaluator did not score exactly both initial and frozen selected checkpoints")
    return protocol, manifest, fits, evidence, decision, p0, evaluation, proof, evaluation_root


def build(outputs, repo):
    outputs, repo = Path(outputs).resolve(), Path(repo).resolve()
    protocol, manifest, fits, evidence, decision, p0, evaluation, eval_proof, eval_root = collect(outputs)
    gates = {key: learning_gates(receipt) for key, receipt in fits.items()}
    total_updates = sum(receipt["optimizer_updates"] for receipt in fits.values())
    total_seconds = sum(receipt["elapsed_seconds"] for receipt in fits.values())
    local_log = outputs / "evidence/local-tests.txt"
    test_text = local_log.read_text() if local_log.exists() else "Local test log unavailable."
    test_summary = re.findall(r"\d+ passed[^\n]*", test_text)
    test_summary = test_summary[-1] if test_summary else "See evidence/local-tests.txt; no test count inferred."
    audit_root = repo / "research/r6/reports"
    files = {}
    files["R6_DATASET_MANIFEST.json"] = json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    files["R6_ARCHITECTURE_AUDIT.md"] = (audit_root / "architecture_audit.md").read_text() + "\n## Executed scope\n\n" + (
        f"Two same-seed native fits actually completed {total_updates} updates. No recurrent comparator or pretrained model was fitted. "
        "Initial and frozen development-selected checkpoints were subsequently scored in a separate CPU execution. "
        "This does not establish graph superiority or multi-seed robustness.\n")
    files["R6_PHYSICS_REGRESSION_ANALYSIS.md"] = (audit_root / "physics_regression.md").read_text() + (
        "\n## R6 correction and comparability\n\nThe preserved pre-final decision is CORRECTION_DECISION_BEFORE_FINAL.json. "
        "It diagnoses loss conditioning from training/development only and authorizes training-only per-axis target standardization within the original budget. "
        "It does not substitute analytical physics at inference. R6 predicts four state components after multiple force stages; legacy predicts a two-component velocity delta. "
        "Their MSE values are not a matched benchmark and must not be subtracted as model improvements.\n\n" + _json_block(decision))
    training = ["# R6 cloud training results", "", f"Two completed FP32 CUDA fits; **{total_updates} actual optimizer updates**. "
                f"Measured fitting/measurement time totals {_number(total_seconds)} seconds ({total_seconds/3600:.6f} device-hours); "
                "this excludes provisioning and is not the provider's billed/quota wall-clock total. Two T4 devices are separate memories; each fit selected cuda:0.", "",
                "| Fit | Updates | Best dev step | Seconds | Parameters | Peak allocated bytes | Selected checkpoint SHA256 |",
                "|---|---:|---:|---:|---:|---:|---|"]
    for key, receipt in fits.items():
        training.append(f"| {key} | {receipt['optimizer_updates']} | {receipt['best_development_step']} | {_number(receipt['elapsed_seconds'])} | {receipt['parameter_count']} | {receipt['peak_cuda_allocated_bytes']} | {receipt['selected_checkpoint_sha256']} |")
    training += ["", "Both runs executed 600 updates under the same predeclared development-selection policy. The selected raw checkpoint is step 400; the standardized checkpoint is step 600. Equal run budgets therefore do not imply equal selected-checkpoint training steps. This comparison is a one-seed conditioning experiment, not an architecture ranking.", "", "## Training/development diagnostics", "", "These objective values belong to each fit's own loss scaling; raw and standardized total objectives are not comparable across fits.", ""]
    for key, receipt in fits.items():
        train_metrics = receipt["final_training"]["native_autoregressive_metrics"]
        dev_metrics = receipt["final_development"]["native_autoregressive_metrics"]
        gaps = {domain: dev_metrics[domain]["exact_accuracy"] - train_metrics[domain]["exact_accuracy"]
                for domain in ("mathematics", "coding")}
        gaps["physics_mse_dev_minus_train"] = dev_metrics["physics"]["mse"] - train_metrics["physics"]["mse"]
        training += ["### " + key, "", "Training receipt: " + key + "/fit/receipt.json. Provider: " + evidence[key]["kernel_id"] + ".", "",
                     _json_block({"loss_summary": receipt["loss_summary"], "initial_training": receipt["initial_training"],
                                  "final_training": receipt["final_training"], "initial_development": receipt["initial_development"],
                                  "selected_development": selected_development(receipt), "final_development": receipt["final_development"],
                                  "final_update_600_dev_minus_train_gaps": gaps,
                                  "gate_2": gates[key], "hardware": receipt["hardware"],
                                  "train_sha256": receipt["training_sha256"], "dev_sha256": receipt["development_sha256"],
                                  "provider_source_sha256": evidence[key]["source_sha256"]})]
    training += ["", "Per-seed variance: **unavailable**. Both fits use seed 61017 and differ only in target normalization; they are not two independent seeds.", "",
                 "P0 used an existing R5.7c checkpoint with zero optimizer updates:", "", _json_block(p0)]
    files["R6_CLOUD_TRAINING_RESULTS.md"] = "\n".join(training) + "\n"
    table = ["| Fit / checkpoint | Regime | Math exact | Coding exact | Physics MSE | Physics MAE |",
             "|---|---|---:|---:|---:|---:|"]
    for entry in evaluation["evaluations"]:
        for split in ("final_iid", "final_ood"):
            metrics = entry["metrics"]["per_split"]
            table.append(f"| {entry['fit_id']} / {entry['role']} | {split} | {_number(metrics[split+'/mathematics']['exact_accuracy'])} | {_number(metrics[split+'/coding']['exact_accuracy'])} | {_number(metrics[split+'/physics']['mse'])} | {_number(metrics[split+'/physics']['mae'])} |")
    failure_analysis, diversity = failure_interpretation(evaluation, eval_root)
    evaluation_text = ["# R6 independent evaluation", "", "Completed separate CPU execution with **zero optimizer updates**. "
                       "Both initial and development-selected checkpoints were frozen before protected final scoring. No further fitting is authorized after these scores.", "",
                       "\n".join(table), "", "Physics aggregate MSE/MAE mixes velocity and position units and is descriptive. Per-axis errors below are primary. "
                       "Program output generation has no answer candidates; candidate-order invariance is not applicable.", "",
                       "## Actual isolation", "", protocol["isolation_scope"], "", _json_block(evaluation["runtime_audit"]), "",
                       "## External baselines", "", "Exact mathematical algorithms, physical solutions and the bounded program executor are external references, not native neural predictions. "
                       "Training-majority/mean baselines were fixed using training data only.", "",
                       _json_block({"training_prior_baselines": evaluation["training_prior_baselines"],
                                    "external_exact_reference_baselines": evaluation["external_exact_reference_baselines"]}), "",
                       "## Reproducibility", "", _json_block({"controller": eval_proof, "final_data_sha256": evaluation["dataset_sha256"],
                                                              "evaluation_manifest_sha256": evaluation["evaluation_manifest_sha256"],
                                                              "prediction_artifact_sha256": evaluation["artifacts"]})]
    files["R6_INDEPENDENT_EVALUATION.md"] = "\n".join(evaluation_text) + "\n"
    general = ["# R6 generalization and failure analysis", "", "\n".join(table), "",
               failure_analysis, "", "Full computed output diversity:", "", _json_block(diversity), "",
               "IID contains unseen canonical contexts from shared template families. OOD tests held-out operand ranges/compositions, three-stage motion, and unseen program control structures. "
               "One seed cannot demonstrate systematic generalization; Gate 3 remains incomplete regardless of these point estimates. "
               "No algebra, matrices, derivatives, circuits, SQL or representative Class 12 benchmark was executed.", ""]
    for entry in evaluation["evaluations"]:
        general += ["## " + entry["fit_id"] + " / " + entry["role"], "",
                    "Raw predictions and failures: `independent-evaluation/downloads/r6-evaluation-output/" + entry["prediction_file"] + "` and `" + entry["failure_file"] + "`.", "",
                    _json_block({"per_topic": entry["metrics"]["per_topic"], "per_split": entry["metrics"]["per_split"],
                                 "carry_borrow": entry["metrics"]["math_carry_borrow"], "interventions": entry["perturbation_summary"]}),
                    "A changed prediction alone does not satisfy a physical law. Intervention response MSE compares the neural state change against the independently verified physical response. "
                    "Identifier-renaming/reformatting invariance with low exact accuracy can describe consistently wrong answers, so both quantities are disclosed.", ""]
        failures = read(eval_root / entry["failure_file"])
        examples = {domain: [row for row in failures if row["domain"] == domain][:3]
                    for domain in ("mathematics", "physics", "coding")}
        general += ["Example failures (up to three per domain):", "", _json_block(examples)]
    files["R6_GENERALIZATION_REPORT.md"] = "\n".join(general) + "\n"
    passed = [(key, domain) for key, domains in gates.items() for domain, result in domains.items() if result["passed"]]
    next_text = ["# R6 next experiment decision", "", "The authorized 1,200-update sprint is complete. No further fit is scheduled or authorized.", "",
                 "Gate 1: completed technical CUDA/checkpoint/source/data verification. Gate 2 is domain-specific, as recorded below. "
                 "Gate 3: incomplete because independent seeds are absent. Gate 4: no curriculum promotion. Gate 5: unexecuted; Class 12 competence is not demonstrated.", "", _json_block(gates), ""]
    if passed:
        next_text += ["Only these fit/domain combinations meet the locked development learning gate: " + ", ".join(key + "/" + domain for key, domain in passed) + ". "
                      "Preserve them as candidates, not accepted general reasoning systems. Recommend one additional bounded validation experiment of at most 600 updates on a new independent seed for the promising component, "
                      "with fresh protected cases and the same task/compute accounting. This is outside the completed budget and requires user review and approval before execution.", ""]
    else:
        next_text += ["No component meets the locked learning gate. Do not expand the curriculum. Recommend a single at-most-600-update diagnostic experiment isolating the foundation output objective "
                      "and measuring free-running exact accuracy against initialization and training priors. Its concrete configuration must be reviewed before execution; this proposed additional budget is UNAPPROVED.", ""]
    next_text += ["Final scores were disclosed once and must not become a repeated tuning set. Any future correction uses development evidence and fresh final cases. "
                  "Lower teacher-forced cross entropy or a physics conditioning improvement does not establish symbolic arithmetic or program semantics. "
                  "Matched recurrent architecture testing and all harder curriculum stages remain unexecuted and unapproved.", "",
                  failure_analysis,
                  "Mathematics/coding representation and decoding changes require a separate future investigation. "
                  "Do not silently add epochs or treat repeated strings as learned execution. The immediate proposed additional physics validation remains bounded and unapproved."]
    files["R6_NEXT_EXPERIMENT_DECISION.md"] = "\n".join(next_text) + "\n"
    files["R6_IMPLEMENTATION_REPORT.md"] = "\n".join([
        "# R6 implementation and execution", "", "Implemented and executed an isolated foundation sprint on branch `research/r6-foundation-20261015`. "
        "Historical frozen R5.7c reference `d54ff8f` and preceding outputs remain outside this experiment namespace.", "",
        "Code areas: `research/r6/datasets`, `architectures`, `training`, `evaluation`, `cloud`, `configs`, `tests`, and `reports`. "
        "Original generators cover signed multi-digit arithmetic, multistage SI motion and bounded Python function/loop tracing. "
        "Safe JSON checkpoints, CUDA-only fitting, exact optimizer accounting, development selection, training-only target normalization, private cloud orchestration and separate CPU scoring are implemented.", "",
        f"Local static/data/inference verification: {test_summary}. No optimizer-based fitting occurred on the Mac. "
        f"Cloud: two completed native fits, {total_updates} optimizer updates, {_number(total_seconds)} measured seconds. P0 reused an existing checkpoint with zero updates. "
        "Separate protected CPU evaluation completed with zero updates, saved predictions and failure records.", "",
        "Dataset: " + str(manifest["validation"]["records_verified"]) + " independently verified original records. "
        "R6 trainer data inputs were restricted to train/development/normalization; R6 final hashes/seeds and R6 generators were excluded from trainer payload. "
        "Frozen legacy source includes old data/evaluation modules for the existing-checkpoint P0 verification; no claim that every historical generator module was excluded is made. "
        "The account owner/controller remains trusted; this is not separately owned evaluation access control.", "",
        "Read R6_CLOUD_TRAINING_RESULTS.md for checkpoint identities/learning gates, R6_INDEPENDENT_EVALUATION.md for actual scores and isolation, "
        "and R6_GENERALIZATION_REPORT.md for per-topic, per-axis and intervention failures. Historical R5.7c heads are incompatible with R6 problem/output contracts; "
        "there is no fabricated zero score or unmatched MSE ranking. No pretrained model was used. All Class 12 specific stages remain unexecuted.", "",
        "The sprint's engineering and cloud execution are complete; competence gates remain evidence-limited. Expanded GPU spending, public publication and future training are not authorized.", ""])
    prototype_search = outputs / "evidence/cbse-prototype-search.txt"
    files["R6_IMPLEMENTATION_REPORT.md"] += ("\n## Existing CBSE prototype search\n\n"
        "The prior CBSE artifact located was a proposal; no executable CBSE upgrade prototype was located in the recorded search. "
        "Search scope and findings are recorded in `evidence/cbse-prototype-search.txt`. This is a search result, not proof that no prototype exists elsewhere.\n"
        if prototype_search.exists() else "\nExisting executable CBSE prototype search evidence is unavailable; no prototype reuse is claimed.\n")
    # Do not replace any final report until all inputs and report construction pass.
    for name, content in files.items():
        (outputs / name).write_text(content)
    return {"status": "REPORTS_BUILT_FROM_VERIFIED_EXECUTION", "reports": sorted(files),
            "optimizer_updates": total_updates, "evaluator_optimizer_updates": 0, "gate_2": gates}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", required=True)
    parser.add_argument("--repo", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.outputs, args.repo), sort_keys=True))


if __name__ == "__main__":
    main()
