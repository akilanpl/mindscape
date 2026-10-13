"""No neural execution: fresh cell balance, metric masks, boundaries and frozen gate logic."""

import copy
import json
from pathlib import Path

import pytest

from mindscape.r5_1.data import build, canonical
from mindscape.r5_1.metrics import aggregate, baselines, summarize
from mindscape.r5_1.qualification import cohort
from mindscape.v5.data import graph
from mindscape.v5.schema import validate


def protocol():
    return json.loads(Path("configs/r5_1/protocol.json").read_text())


def test_deterministic_fresh_manifest_and_operator_validity_cartesian_balance():
    p = protocol()
    m = build(p)
    assert build(p) == m and len(m["tasks"]) == 108
    old = {canonical(t) for t in json.loads(Path("configs/v5/tasks.json").read_text())["tasks"]}
    assert not old & {canonical(t) for t in m["tasks"]}
    for split in ("development", "qualification", "composition"):
        coding = [t for t in m["tasks"] if t["domain"] == "coding" and t["split"] == split]
        expected = (4, 2) if split == "development" else (2, 1)
        for op in ("add", "sub", "mul"):
            subset = [t for t in coding if t["public"]["operations"][0] == op]
            assert (
                sum(t["public"]["target_leaf"] < len(t["public"]["constants"]) for t in subset)
                == expected[0]
            )
            assert (
                sum(t["public"]["target_leaf"] >= len(t["public"]["constants"]) for t in subset)
                == expected[1]
            )


def test_changed_entities_average_all_coordinates_and_invalid_remains_separate():
    target = {"delta": [[0.1, 0.0], [0.0, 0.0]], "valid": 1}
    out = summarize(target, [[0.0, 0.0], [0.0, 0.0]])
    assert out["whole_graph_mse"] == pytest.approx(0.0025)
    assert out["changed_entity_mse"] == pytest.approx(0.005)
    assert out["unchanged_entity_mse"] == 0
    invalid = summarize({"delta": [[0.0, 0.0]], "valid": 0}, [[0.5, 0.0]])
    assert invalid["changed_entity_mse"] is None and invalid["whole_graph_mse"] == 0.125
    assert aggregate([out, invalid])["changed_entity_mse"] == out["changed_entity_mse"]
    assert aggregate([out, invalid])["whole_graph_mse"] > out["whole_graph_mse"]


def test_zero_effect_valid_task_not_invented_changed_success():
    out = summarize({"delta": [[0.0]], "valid": 1}, [[1.0]])
    assert out["changed_entities"] == 0 and out["changed_entity_mse"] is None
    assert out["unchanged_entity_mse"] == 1


def test_graph_boundary_mask_still_rejects_hidden_and_target_fields():
    task = build(protocol())["tasks"][0]
    g = graph(task)
    with pytest.raises(ValueError):
        validate({**g, "target": [999]})
    bad = copy.deepcopy(g)
    bad["observed"][0][0] = 0
    with pytest.raises(ValueError):
        validate(bad)
    for task in build(protocol())["tasks"]:
        validate(graph(task))


def test_baseline_uses_no_target_validity_or_effect_to_predict():
    task = {"domain": "coding", "public": {"constants": [13], "target_leaf": 0, "increment": 2}}
    actual = {"delta": [[0.0], [0.02], [0.02]], "valid": 1}
    metrics = baselines(task, actual)
    assert metrics["persistence"]["changed_entity_mse"] == pytest.approx(0.0004)
    assert metrics["local_action"]["changed_entity_mse"] == pytest.approx(0.0002)
    # Outcome validation changes can change metric labels, not how the simple public-action rule behaves.
    actual["valid"] = 0
    assert (
        baselines(task, actual)["local_action"]["changed_entity_mse"]
        == metrics["local_action"]["changed_entity_mse"]
    )


def test_no_effect_or_all_negative_cohort_fails_fixed_gates():
    rows = [
        {
            "domain": "mathematics",
            "split": "qualification",
            "baselines": {
                "persistence": summarize({"delta": [[0.0]], "valid": 0}, [[0.0]]),
                "local_action": summarize({"delta": [[0.0]], "valid": 0}, [[0.0]]),
            },
            "target": {"delta": [[0.0]], "valid": 0},
        }
        for _ in range(9)
    ]
    result = cohort(rows, "mathematics", "qualification", protocol())
    assert result["pass"] is False and result["checks"]["valid"] is False
    assert result["checks"]["persistence_headroom"] is False


def test_shape_declarations_match_unchanged_v5_schema():
    from mindscape.v5.schema import DIMS, MAX_NODES, OUTPUTS

    declared = protocol()["engineering"]
    assert declared["encoder_feature_widths"] == DIMS
    assert declared["effect_widths"] == OUTPUTS
    assert declared["node_padding"] == MAX_NODES


def test_runner_contains_no_optimizer_or_training_entrypoint():
    import ast

    for path in Path("src/mindscape/r5_1").glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    assert node.func.attr not in ("step", "fit", "train", "save_checkpoint")
                if isinstance(node.func, ast.Name):
                    assert node.func.id not in ("Adam", "SGD", "fit")
