import copy
import json
from pathlib import Path

import pytest

from mindscape.v4.data import DOMAINS, tasks
from mindscape.v4.encoding import DIMS, encode
from mindscape.v4.verify import outcomes


def test_fresh_split_identities_and_sizes():
    rows = tasks()
    assert len(rows) == len({r["identity"] for r in rows}) == 60
    for domain in DOMAINS:
        assert [
            sum(r["domain"] == domain and r["split"] == s for r in rows)
            for s in ("train", "iid", "ood")
        ] == [8, 4, 8]
    assert tasks() == rows


def test_math_wrong_transition_not_repaired():
    task = {
        "domain": "mathematics",
        "public": {"operation": "add", "left": 7, "right": 8, "incoming": 1},
    }
    good = {
        "claim": {
            "operation": 0,
            "raw": 16,
            "digit": 6,
            "carry": 1,
            "next_digit": 6,
            "next_carry": 1,
        }
    }
    bad = copy.deepcopy(good)
    bad["claim"]["next_digit"] = 3
    result = outcomes(task, [bad, good])
    assert result[0] == {"success": False, "written_digit": 3, "written_carry": 1}
    assert result[1]["success"]
    assert len(encode("mathematics", task["public"], good, 0.5)) == DIMS["mathematics"]


def test_private_fields_rejected():
    task = next(t for t in tasks() if t["domain"] == "mathematics")
    proposal = {
        "claim": {
            "operation": 0,
            "raw": 1,
            "digit": 1,
            "carry": 0,
            "next_digit": 1,
            "next_carry": 0,
        }
    }
    for name in ("private", "labels", "expected", "future", "split", "identity"):
        with pytest.raises(ValueError):
            encode("mathematics", {**task["public"], name: 1}, proposal, 0)
    with pytest.raises(ValueError):
        encode("mathematics", task["public"], {**proposal, "success": True}, 0)


def test_coding_literals_and_private_independence():
    task = next(t for t in tasks() if t["domain"] == "coding")
    public = {**task["public"], "actual_visible_result": {}}
    proposal = {"repository": {"solution.py": "def f(v): return sum(x for x in v if x >= 21)"}}
    x = encode("coding", public, proposal, 0)
    other = {"repository": {"solution.py": "def f(v): return sum(x for x in v if x >= 22)"}}
    assert x != encode("coding", public, other, 0)
    task["private"] = {"expected": "poison"}
    assert x == encode("coding", public, proposal, 0)
    with pytest.raises(ValueError):
        encode("coding", {**public, "private": task["private"]}, proposal, 0)


def test_physics_reset_and_action_specific_outcome():
    task = next(t for t in tasks() if t["domain"] == "physics")
    zero = [[0.0] * 4 for _ in range(3)]
    push = copy.deepcopy(zero)
    push[0][2] = 2.0
    result = outcomes(task, [{"action": zero}, {"action": push}, {"action": zero}])
    assert result[0] == result[2]
    assert result[0]["terminal"] != result[1]["terminal"]


def test_protocol_is_bounded_and_no_holdout_reads():
    protocol = json.loads(Path("configs/v4/protocol.json").read_text())
    assert protocol["seeds"] == [11, 23, 47]
    assert protocol["pretrain_updates"] + protocol["target_updates"] == 120
    assert protocol["target_training_tasks"] == 4


def test_metrics_include_unsolvable_tasks_and_calibration():
    from mindscape.v4.scoring import metrics

    rows = [{"identity": "a", "labels": [0, 1]}, {"identity": "b", "labels": [0, 0]}]
    predictions = [
        {"identity": "a", "selected": 1, "probabilities": [0.1, 0.8]},
        {"identity": "b", "selected": 0, "probabilities": [0.2, 0.3]},
    ]
    result = metrics(rows, predictions)
    assert result["accuracy"] == 0.5
    assert result["conditional_accuracy"] == 1
    assert result["oracle_accuracy"] == 0.5
    assert result["brier"] == pytest.approx(0.045)


def test_actual_wasi_repair_and_near_miss():
    task = next(t for t in tasks() if t["domain"] == "coding")
    original = task["public"]["repository"]
    repaired = {k: v.replace(" > ", " >= ") for k, v in original.items()}
    result = outcomes(task, [{"repository": original}, {"repository": repaired}])
    assert [r["success"] for r in result] == [False, True]
    assert all(r["actual_vm_calls"] == 2 for r in result)


def test_collection_persists_features_before_outcomes(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from mindscape.v4 import collect, verify

    task = next(t for t in tasks() if t["domain"] == "mathematics")
    proposal = {
        "claim": {
            "operation": 0,
            "raw": 2,
            "digit": 2,
            "carry": 0,
            "next_digit": 2,
            "next_carry": 0,
        }
    }
    monkeypatch.setattr(
        collect, "proposals", lambda t, a: (t["public"], [proposal], [0.5], {"test_fixture": True})
    )

    def scorer(t, candidates, executor):
        saved = json.loads((tmp_path / "proposals" / f"{t['identity']}.json").read_text())
        assert "features" in saved
        assert not {"labels", "outcomes", "private"} & saved.keys()
        return [{"success": True}]

    monkeypatch.setattr(verify, "outcomes", scorer)
    rows = collect.collect([task], SimpleNamespace(loaded={}), tmp_path)
    assert rows[0]["labels"] == [1.0]


def test_replay_rejects_changed_labels(tmp_path, monkeypatch):
    from mindscape.v3.schema import digest
    from mindscape.v4 import data, verify

    task = next(t for t in tasks() if t["domain"] == "mathematics")
    proposal = {
        "claim": {
            "operation": 0,
            "raw": 2,
            "digit": 2,
            "carry": 0,
            "next_digit": 2,
            "next_carry": 0,
        }
    }
    saved = {
        "identity": task["identity"],
        "domain": task["domain"],
        "split": task["split"],
        "public": task["public"],
        "proposals": [proposal],
        "scores": [0],
        "features": [encode(task["domain"], task["public"], proposal, 0)],
    }
    (tmp_path / "proposals").mkdir()
    (tmp_path / "proposals" / f"{task['identity']}.json").write_text(json.dumps(saved))
    row = {
        **saved,
        "proposal_digest": digest(saved),
        "outcomes": outcomes(task, [proposal]),
        "labels": [0.0],
    }
    (tmp_path / "rows.jsonl").write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(data, "tasks", lambda: [task])
    with pytest.raises(ValueError, match="Label mismatch"):
        verify.replay(tmp_path)
