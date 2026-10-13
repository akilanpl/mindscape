"""Nonneural qualification guards; fabricated rows below are unit fixtures, not results."""

import json
from pathlib import Path

import pytest

from mindscape.v4_qualification.freeze import MANIFEST, PROPOSAL, build, canonical_math
from mindscape.v4_qualification.gates import decision, evaluate
from mindscape.v4_qualification.run import normalize, verify_prefix


def rows_for(domain, split, labels):
    start = 0 if split == "train" else 8
    return [
        {
            "domain": domain,
            "split": split,
            "index": start + i,
            "identity": f"{domain}-{split}-{i}",
            "labels": y,
            "scores": list(range(len(y))),
        }
        for i, y in enumerate(labels)
    ]


def test_manifest_is_deterministic_fresh_balanced_and_capped():
    from mindscape.v4.data import tasks

    frozen = json.loads(MANIFEST.read_text())
    assert build() == frozen
    assert len(frozen["tasks"]) == 36
    old = {canonical_math(t["public"]) for t in tasks() if t["domain"] == "mathematics"}
    for split, count in [("train", 8), ("qualification", 4)]:
        selected = [
            t for t in frozen["tasks"] if t["domain"] == "mathematics" and t["split"] == split
        ]
        assert len(selected) == count
        assert sum(t["public"]["incoming"] for t in selected) == count // 2
        assert not old & {canonical_math(t["public"]) for t in selected}


def test_incomplete_gate_never_passes_and_four_failure_stops():
    p = json.loads(PROPOSAL.read_text())
    rows = rows_for("physics", "train", [[0] * 9] * 4)
    assert decision(evaluate(rows[:3], p), 3) == "PENDING"
    gates = evaluate(rows, p)
    assert gates["physics.target_training"]["positive"] == 0
    assert gates["physics.target_training"]["negative"] == 36
    assert decision(gates, 4) == "DATA_NO_GO"


def test_original_thresholds_and_headroom_enforced():
    p = json.loads(PROPOSAL.read_text())
    rows = []
    for d in ("mathematics", "physics", "coding"):
        rows += rows_for(d, "train", [[0, 1]] * 4)
        rows += rows_for(d, "qualification", [[0, 1]] * 4)
    gates = evaluate(rows, p)
    assert all(gates[d + ".target_training"]["pass"] for d in ("mathematics", "physics", "coding"))
    assert gates["coding.headroom"]["oracle_minus_specialist"] == 0
    assert gates["coding.headroom"]["pass"] is False
    # Leave one coding task's specialist wrong, while first-candidate stays wrong elsewhere.
    rows[-1]["scores"] = [2, 1]
    gates = evaluate(rows, p)
    assert gates["coding.headroom"]["pass"] is True
    assert decision(gates, 24) == "PENDING"
    assert decision(gates, 36) == "DATA_READY"


def test_no_request_allowed_after_failed_cohort():
    p = json.loads(PROPOSAL.read_text())
    rows = rows_for("mathematics", "train", [[0] * 4] * 4)
    rows += rows_for("physics", "train", [[0] * 9])
    tasks = [{"identity": r["identity"]} for r in rows]
    with pytest.raises(ValueError, match="continued"):
        verify_prefix(tasks, rows, p)


def test_ast_dedup_preserves_best_score_and_shuffle_is_deterministic():
    task = {"domain": "coding", "identity": "a" * 64}
    candidates = [
        {"repository": {"solution.py": "def f(x): return x\n"}},
        {"repository": {"solution.py": "def f(x):\n    return x\n"}},
        {"repository": {"solution.py": "def f(x): return x+1\n"}},
    ]
    a = normalize(task, candidates, [0.1, 0.9, 0.5])
    assert a == normalize(task, candidates, [0.1, 0.9, 0.5])
    assert len(a) == 2
    group = next(x for x in a if len(x["source_indices"]) == 2)
    assert group["score"] == 0.9 and group["source_indices"] == [0, 1]
    with pytest.raises(ValueError, match="budget"):
        normalize(task, candidates + [candidates[0]], [0] * 4)


def test_no_training_entrypoint_or_optimizer_calls():
    import ast

    tree = ast.parse(Path("src/mindscape/v4_qualification/run.py").read_text())
    forbidden = {"fit", "backward", "step", "save_checkpoint", "train"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in forbidden


def test_actual_arithmetic_replay_and_label_corruption(tmp_path):
    from mindscape.v3.schema import digest
    from mindscape.v4.encoding import encode
    from mindscape.v4.verify import outcomes
    from mindscape.v4_qualification.run import save, verify_row

    task = {
        "domain": "mathematics",
        "split": "train",
        "index": 0,
        "identity": "fixture",
        "public": {"operation": "add", "left": 6, "right": 8, "incoming": 1},
        "scorer": {"digit": 5, "carry": 1},
    }
    p = {
        "claim": {
            "operation": 0,
            "raw": 0,
            "digit": 0,
            "carry": 0,
            "next_digit": 0,
            "next_carry": 0,
        }
    }
    raw = {
        "public": task["public"],
        "candidates": [p],
        "scores": [0.5],
        "provenance": {"unit_fixture": True},
    }
    groups = normalize(task, [p], [0.5])
    (tmp_path / "raw").mkdir()
    (tmp_path / "proposals").mkdir()
    save(tmp_path / "raw" / "fixture.json", raw)
    row = {k: task[k] for k in ("domain", "split", "index", "identity")}
    row.update(
        public=task["public"],
        proposals=[p],
        scores=[0.5],
        normalized=groups,
        raw_digest=digest(raw),
        features=[encode("mathematics", task["public"], p, 0.5)],
    )
    save(tmp_path / "proposals" / "fixture.json", row)
    row["proposal_digest"] = digest(row.copy())
    row["outcomes"] = outcomes(task, [p])
    row["labels"] = [0]
    verify_row(task, row, tmp_path)
    row["labels"] = [1]
    with pytest.raises(ValueError, match="Labels"):
        verify_row(task, row, tmp_path)
