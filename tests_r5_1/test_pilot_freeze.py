"""Specification integrity only; no composition execution, optimizer or neural calls."""

import hashlib
import json
from pathlib import Path


def test_pilot_is_bounded_and_explicitly_unapproved():
    root = Path("configs/r5_1/pilot_v1")
    p = json.loads((root / "protocol.json").read_text())
    binding = json.loads((root / "data-binding.json").read_text())
    assert p["status"] == "FROZEN_AWAITING_EXPLICIT_AUTHORIZATION"
    assert p["fits"] == len(p["optimizer_seeds"]) * 3 == 9
    assert (
        p["graph_presentations_per_fit"] == p["maximum_updates_per_fit"] * p["batch_size"] == 1920
    )
    assert p["total_maximum_updates"] == p["fits"] * p["maximum_updates_per_fit"] == 1080
    assert p["total_maximum_graph_presentations"] == 17280
    assert (
        binding["protocol_sha256"]
        == hashlib.sha256((root / "protocol.json").read_bytes()).hexdigest()
    )
    assert (
        binding["training_executed"] is False and binding["composition_outputs_generated"] is False
    )


def test_pilot_labels_and_public_evaluation_are_disjoint():
    b = json.loads(Path("configs/r5_1/pilot_v1/data-binding.json").read_text())
    train = {v for values in b["development_identities"].values() for v in values}
    evaluation = {t["identity"] for t in b["composition_coding_tasks"]}
    assert len(train) == 54 and len(evaluation) == 9 and not train & evaluation
    assert all(
        t["domain"] == "coding" and t["split"] == "composition"
        for t in b["composition_coding_tasks"]
    )
    assert all("target" not in t and "delta" not in t for t in b["composition_coding_tasks"])
    assert all(len(v) == 18 for v in b["development_identities"].values())


def test_pilot_remains_bound_to_preserved_readiness_data():
    b = json.loads(Path("configs/r5_1/pilot_v1/data-binding.json").read_text())
    r = Path("results/r5_1/screen-2cac9d0fcef076df")
    assert (
        hashlib.sha256((r / "rows.json").read_bytes()).hexdigest() == b["development_rows_sha256"]
    )
    assert (
        hashlib.sha256((r / "manifest.json").read_bytes()).hexdigest()
        == b["readiness_manifest_sha256"]
    )
    assert (
        hashlib.sha256((r.with_suffix(".tar.gz")).read_bytes()).hexdigest()
        == b["readiness_evidence_archive_sha256"]
    )
