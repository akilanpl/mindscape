"""Focused nonneural semantic boundaries and independent execution controls."""

import copy
import json
from pathlib import Path

import pytest

from mindscape.v5.data import graph, public_before, tasks
from mindscape.v5.execute import execute
from mindscape.v5.qualify import check
from mindscape.v5.schema import validate


def protocol():
    return json.loads(Path("configs/v5/protocol.json").read_text())


def test_disjoint_budget_and_compositions():
    rows = tasks(protocol())
    assert len(rows) == len({r["identity"] for r in rows}) == 132
    for d in ("mathematics", "coding", "physics"):
        assert [
            sum(r["domain"] == d and r["split"] == s for r in rows)
            for s in ("train", "qualification", "composition")
        ] == [24, 8, 12]
    assert rows == tasks(protocol())
    for r in rows:
        validate(graph(r))


def test_private_target_and_hidden_values_rejected():
    task = tasks(protocol())[0]
    g = graph(task)
    for key in ("target", "after", "valid", "split", "identity", "private"):
        with pytest.raises(ValueError):
            validate({**g, key: 0})
    bad = copy.deepcopy(g)
    bad["observed"][0][0] = 0
    with pytest.raises(ValueError, match="Unobserved"):
        validate(bad)
    bad["entities"][0][0] = 0
    validate(bad)


def test_math_changes_and_invalid_noop_are_executed():
    rows = [r for r in tasks(protocol()) if r["domain"] == "mathematics"]
    first = execute(rows[0])
    invalid = execute(rows[2])
    assert first["valid"] == 1 and first["after"] != first["before"]
    assert invalid["valid"] == 0 and invalid["after"] == invalid["before"]
    assert execute(rows[0]) == first


def test_coding_actual_wasi_effect_and_literal_nodes():
    task = next(
        r for r in tasks(protocol()) if r["domain"] == "coding" and r["split"] == "composition"
    )
    result = execute(task)
    assert result["provenance"]["executor"] == "actual_WASI"
    assert result["provenance"]["calls"] == 2
    assert result["before"] == public_before(task)
    assert result["after"] != result["before"]
    g = graph(task)
    assert g["entities"][3][0] == g["entities"][4][0] == 0
    assert g["observed"][3][0] == g["observed"][4][0] == 0
    before = copy.deepcopy(g)
    result["after"] = [[999]]
    assert graph(task) == before


def test_physics_public_impulse_independent_replay():
    task = next(r for r in tasks(protocol()) if r["domain"] == "physics")
    out = execute(task)
    p = task["public"]
    v = out["after"][0]
    assert v[2] == pytest.approx(p["velocities"][0][0] + p["impulse"][0] / p["masses"][0], rel=1e-6)
    assert execute(task) == out
    assert graph(task)["observed"] == [[1] * 8]


def test_gate_cannot_pass_all_negative_or_unchanged():
    rows = [
        {"domain": "coding", "split": "train", "target": {"valid": 0, "delta": [[0.0]]}}
        for _ in range(12)
    ]
    assert check(rows, "coding", "train", protocol())["pass"] is False
    for r in rows:
        r["target"]["valid"] = 1
    assert check(rows, "coding", "train", protocol())["pass"] is False


def test_architecture_contains_shared_effect_generator_and_domain_heads():
    # Source structure guard only; learned behavior is checked on native MPS, never faked on CPU.
    import ast

    source = Path("src/mindscape/v5/model.py").read_text()
    tree = ast.parse(source)
    assert {n.name for n in tree.body if isinstance(n, ast.ClassDef)} == {
        "TransitionCore",
        "GraphTransition",
    }
    assert "self.core = TransitionCore()" in source
    assert "self.heads" in source and "self.encoders" in source
    assert "target" not in {n.arg for n in ast.walk(tree) if isinstance(n, ast.arg)}


def test_data_no_go_blocks_native_training_before_device_access():
    from mindscape.v5.__main__ import main

    with pytest.raises(RuntimeError, match="training forbidden"):
        main()
    assert not Path("results/v5/native-pilot").exists()


def test_json_provenance_replay_keeps_exact_numeric_effects():
    from mindscape.v3.schema import digest

    task = next(r for r in tasks(protocol()) if r["domain"] == "physics")
    original = execute(task)
    saved = json.loads(json.dumps(original))
    assert digest(original) == digest(saved)
    assert original["delta"] == saved["delta"]
