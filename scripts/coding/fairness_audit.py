"""Automated boundaries and matched-training identity checks; no causal shortcut."""

import json
from pathlib import Path

from mindscape.coding.schema import CodeTask

root = Path("results/coding/completion_audits_v1")
root.mkdir(parents=True, exist_ok=True)
dataset = json.loads(Path("results/coding/final_dataset_v1/dataset.json").read_text())
checks = []


def check(name, value, detail=""):
    checks.append({"name": name, "pass": bool(value), "detail": detail})


for split, values in dataset.items():
    for value in values:
        t = CodeTask(**value)
        view = t.visible()
        check(
            "private_field_exclusion:" + t.task_id,
            not {"hidden_tests", "correct_repository", "ground_truth_patch", "metadata"}
            & set(view),
        )
        check("view_repository_is_buggy_input:" + t.task_id, view["repository"] == t.repository)
train_ids = {t["task_id"] for t in dataset["train"]}
train_seeds = {t["metadata"]["seed"] for t in dataset["train"]}
train_structures = {t["metadata"]["structural_category"] for t in dataset["train"]}
for split in ("validation", "test", "ood_test"):
    check("no_task_overlap:" + split, not train_ids & {t["task_id"] for t in dataset[split]})
    check(
        "no_seed_overlap:" + split,
        not train_seeds & {t["metadata"]["seed"] for t in dataset[split]},
    )
check(
    "structural_ood_holdout",
    not train_structures & {t["metadata"]["structural_category"] for t in dataset["ood_test"]},
)
for path in Path("results/coding/gradient_v1").glob("seed_*/training.json"):
    training = json.loads(path.read_text())
    expected = [t["task_id"] for t in dataset["train"][: training["samples"]]]
    check("matched_patch_training_tasks:" + str(path), training["task_ids"] == expected)
    check(
        "patch_training_private_input_excluded:" + str(path),
        not training["hidden_tests_in_training"],
    )
lockbox = json.loads(Path("results/coding/completion_lockbox_v1/tasks.json").read_text())
old_seeds = {t["metadata"]["seed"] for values in dataset.values() for t in values}
check("lockbox_excludes_all_prior_seeds", not old_seeds & {t["metadata"]["seed"] for t in lockbox})
check("lockbox_has_100_unique_tasks", len({t["task_id"] for t in lockbox}) == 100)
check(
    "lockbox_structural_holdout",
    not train_structures
    & {
        t["metadata"]["structural_category"]
        for t in lockbox
        if t["metadata"]["split"] == "ood_test"
    },
)
for path in Path("results/coding/completion_gradient_v1").glob("*/seed_*/training.json"):
    training = json.loads(path.read_text())
    expected = [t["task_id"] for t in dataset["train"][: training["samples"]]]
    check("matched_training_tasks:" + str(path), training["task_ids"] == expected)
    check(
        "no_hidden_or_future_inputs:" + str(path),
        not training["hidden_inputs"] and not training["future_state_input"],
    )
report = {
    "checks": checks,
    "check_count": len(checks),
    "local_leakage_status": "PASS" if all(x["pass"] for x in checks) else "FAIL",
    "pretraining_contamination": "UNKNOWN; cannot establish absence from public model training",
    "overall_contamination_free_claim": "FAIL / not established",
    "fairness": {
        "shared_backbone": True,
        "matched_unique_task_budgets": True,
        "equal_supervision": False,
        "equal_inference_compute": False,
        "causal_architecture_isolation": "NOT ESTABLISHED",
        "conditions": {
            "model_only": {
                "training": "correct-source patch labels",
                "actions": "single patch",
                "environment_feedback": False,
                "memory": False,
            },
            "structured": {
                "training": "state/relations/goal plus patch labels",
                "actions": "single patch",
                "environment_feedback": False,
                "memory": False,
            },
            "mindscape_b": {
                "training": "accepted expert action labels; explicit state/relations/goal",
                "actions": "up to eight typed actions",
                "environment_feedback": "visible tests",
                "memory": "training-only episodic examples",
            },
            "mindscape_c": {
                "training": "real accepted expert state/action/event/result/next-state replay; pre-action observations only in model input",
                "actions": "up to eight typed actions",
                "environment_feedback": "visible tests",
                "memory": "training-only episodic examples",
            },
        },
    },
    "interpretation": "Any gains are conditional on disclosed supervision and interaction differences; do not attribute them uniquely to architecture.",
}
(root / "fairness_leakage.json").write_text(json.dumps(report, indent=2))
print(report["check_count"], report["local_leakage_status"])
