"""Pure-arithmetic/syntax fixtures only; no model forwards or optimizer execution."""

import ast
import json
from pathlib import Path

import pytest

from mindscape.r5_1_pilot.scoring import baseline_predictions, gate_report, metrics, public_valid


def protocol():
    return json.loads(Path("configs/r5_1/learning_v2/protocol.json").read_text())


def fixture():
    fits = []
    base = {}
    for d in ("mathematics", "coding", "physics"):
        base[d] = {
            "persistence": {"changed_entity_mse": 2.0},
            "local_action": {"changed_entity_mse": 1.5},
        }
        for seed in (11, 23, 47):
            for c, changed, whole in [
                ("domain_only", 1.0, 0.6),
                ("shared", 0.5, 0.3),
                ("randomized_auxiliary", 1.2, 0.7),
            ]:
                fits.append(
                    {
                        "identity": {"target": d, "condition": c, "seed": seed},
                        "composition": {
                            "changed_entity_mse": changed,
                            "whole_graph_mse": whole,
                            "unchanged_entity_mse": 0.00005,
                            "invalid_noop_mse": 0.00005,
                            "validity_balanced_accuracy": 0.9,
                            "finite": True,
                        },
                        "gradient_audit_pass": True,
                        "initial_training": {"changed_entity_mse": 1.0},
                        "final_training": {"changed_entity_mse": 0.5},
                    }
                )
    return fits, base


def test_exact_budget_and_unchanged_architecture_reference():
    p = protocol()
    assert p["fits"] == 27 and p["maximum_updates_per_fit"] == 120 and p["batch_size"] == 16
    assert p["total_maximum_updates"] == 3240 and p["total_maximum_graph_presentations"] == 51840
    assert p["optimizer_seeds"] == [11, 23, 47]
    assert (
        p["scientific_gates_all_required"][
            "maximum_mean_other_domain_relative_changed_mse_regression"
        ]
        == 0.10
    )


def test_gates_require_both_controls_and_other_domain_guard():
    f, b = fixture()
    assert gate_report(f, b, protocol())["decision"] == "GO_TO_CONFIRMATION"
    for fit in f:
        if fit["identity"]["target"] == "mathematics" and fit["identity"]["condition"] == "shared":
            fit["composition"]["changed_entity_mse"] = 1.11
    assert gate_report(f, b, protocol())["gates"]["other_domain_regression"] is False
    f, b = fixture()
    for fit in f:
        if (
            fit["identity"]["target"] == "coding"
            and fit["identity"]["condition"] == "randomized_auxiliary"
        ):
            fit["composition"]["changed_entity_mse"] = 0.4
    result = gate_report(f, b, protocol())
    assert result["decision"] == "NO_GO" and not result["gates"]["primary_both_control_gains"]


def test_zero_control_cannot_produce_epsilon_assisted_gain():
    f, b = fixture()
    for fit in f:
        if fit["identity"]["target"] == "coding" and fit["identity"]["condition"] == "domain_only":
            fit["composition"]["changed_entity_mse"] = 0.0
    out = gate_report(f, b, protocol())
    assert out["relative_changed_mse_gains"]["coding"]["domain_only"] == [None] * 3
    assert not out["gates"]["primary_seed_consistency"]


def test_changed_and_unchanged_metrics_keep_invalid_task_denominators():
    rows = [
        {
            "identity": "a",
            "domain": "coding",
            "target": {"valid": 1, "delta": [[0.0], [0.02], [0.02]]},
        },
        {
            "identity": "b",
            "domain": "coding",
            "target": {"valid": 0, "delta": [[0.0], [0.0], [0.0]]},
        },
    ]
    predictions = [
        {"identity": "a", "delta": [[0.5], [0.02], [0.02]], "valid_probability": 0.9},
        {"identity": "b", "delta": [[0.0], [0.0], [0.0]], "valid_probability": 0.1},
    ]
    m, _d = metrics(rows, predictions)
    assert (
        m["tasks"] == 2 and m["tasks_with_changed_entities"] == 1 and m["changed_entity_mse"] == 0
    )
    assert m["unchanged_entity_mse"] > 0 and m["whole_graph_mse"] > 0
    assert m["invalid_noop_mse"] == 0 and m["validity_balanced_accuracy"] == 1
    with pytest.raises(ValueError):
        metrics(rows, predictions + [predictions[0]])


def test_simple_validity_uses_public_bounds_not_private_outcomes():
    task = {
        "identity": "fixture",
        "domain": "coding",
        "public": {
            "x": 8,
            "constants": [13],
            "operations": ["mul"],
            "target_leaf": 0,
            "increment": 2,
            "entry": "f",
        },
    }
    assert public_valid(task) == 1
    p = baseline_predictions(task)
    assert p["local_action"]["delta"] == [[0.0], [0.02], [0.0]]
    task["public"]["target_leaf"] = 9
    assert public_valid(task) == 0


def test_fit_labels_and_inference_arguments_are_separated():
    source = ast.parse(Path("src/mindscape/r5_1_pilot/learning.py").read_text())
    functions = {n.name: n for n in source.body if isinstance(n, ast.FunctionDef)}
    assert [a.arg for a in functions["predict"].args.args] == ["model", "graphs"]
    calls = [
        n
        for n in ast.walk(functions["fit"])
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "model"
    ]
    assert len(calls) == 1 and not calls[0].keywords
    text = Path("src/mindscape/r5_1_pilot/__main__.py").read_text()
    assert text.index("prediction-freeze.json") < text.index("target = execute(task)")
    assert "exist_ok=False" in text and "resume" not in {
        a.arg for a in ast.walk(source) if isinstance(a, ast.arg)
    }
