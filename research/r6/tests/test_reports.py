"""Pure report guards and threshold checks; no neural execution or fitting."""
import json
import pytest
from research.r6.reports.build_reports import collect, learning_gates, output_diversity, pinned, selected_development


def _receipt(math=.07, coding=.07, physics=75):
    initial = {"mathematics": {"exact_accuracy": 0.}, "coding": {"exact_accuracy": 0.}, "physics": {"mse": 110.}}
    selected = {"mathematics": {"exact_accuracy": math}, "coding": {"exact_accuracy": coding}, "physics": {"mse": physics}}
    return {"best_development_step": 100, "dev_history": [{"update": 100, "native_autoregressive_metrics": selected}],
            "initial_development": {"native_autoregressive_metrics": initial},
            "development_training_only_baseline": {"mathematics": {"exact_accuracy": .01}, "coding": {"exact_accuracy": .01}, "physics": {"mse": 100.}}}


def test_locked_learning_gates_are_domain_specific():
    gates = learning_gates(_receipt())
    assert all(row["passed"] for row in gates.values())
    gates = learning_gates(_receipt(math=.04, coding=.02, physics=90))
    assert not any(row["passed"] for row in gates.values())


def test_initialization_selection_has_no_invented_history():
    receipt = _receipt()
    receipt["best_development_step"] = 0
    assert selected_development(receipt) == receipt["initial_development"]
    assert not any(row["passed"] for row in learning_gates(receipt).values())


def test_missing_selected_development_step_rejected():
    receipt = _receipt()
    receipt["best_development_step"] = 200
    with pytest.raises(ValueError):
        selected_development(receipt)


def test_fail_closed_without_completed_input_receipts(tmp_path):
    with pytest.raises(FileNotFoundError):
        collect(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_hash_pin_rejects_missing_and_traversal(tmp_path):
    with pytest.raises(ValueError):
        pinned(tmp_path, "missing", "0"*64)
    with pytest.raises(ValueError):
        pinned(tmp_path, "../elsewhere", "0"*64)


def test_protocol_budget_changes_rejected_before_report_write(tmp_path):
    (tmp_path/"R6_PROTOCOL_LOCK.json").write_text(json.dumps({"total_optimizer_update_ceiling": 1800, "updates_per_fit": 600}))
    with pytest.raises(ValueError, match="budget"):
        collect(tmp_path)
    assert len(list(tmp_path.iterdir())) == 1


def test_output_diversity_counts_emitted_answers_and_separates_regimes():
    records = [
        {"domain": "mathematics", "split": "final_iid", "target": {"answer": "-12"}, "prediction": {"answer": "-12"}},
        {"domain": "mathematics", "split": "final_ood", "target": {"answer": "42"}, "prediction": {"answer": "-12"}},
        {"domain": "coding", "split": "final_iid", "target": {"answer": "42"}, "prediction": {"answer": None}},
    ]
    result = output_diversity(records)
    assert result["all-final/mathematics"] == {"count": 2, "distinct_valid_answers": 1, "invalid_answers": 0,
        "answer_counts": {"-12": 2}, "most_common_fraction": 1., "exact_correct": 1}
    assert result["final_iid/mathematics"]["exact_correct"] == 1
    assert result["final_ood/mathematics"]["exact_correct"] == 0
    assert result["all-final/coding"]["invalid_answers"] == 1
    assert result["all-final/coding"]["distinct_valid_answers"] == 0


def test_diversity_keeps_zero_and_negative_outputs_distinct():
    rows = [{"domain": "coding", "split": "final_iid", "target": {"answer": "0"}, "prediction": {"answer": answer}}
            for answer in ("0", "-12", "0")]
    result = output_diversity(rows)["all-final/coding"]
    assert result["answer_counts"] == {"0": 2, "-12": 1}
    assert result["most_common_fraction"] == 2/3
    assert result["exact_correct"] == 2
