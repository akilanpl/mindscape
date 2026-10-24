"""Dataset-only integration checks; never import torch or execute fitting."""
import copy
import json
import statistics
import pytest
from research.r6.datasets.build import build, normalization, sha256, validate_splits, verify_row
from research.r6.datasets import coding, mathematics, physics


SMALL_COUNTS = {"train": 12, "dev": 6, "final_iid": 6, "final_ood": 6}


def test_reproducible_new_directory_build_and_protected_allowlist(tmp_path):
    first, second = tmp_path/"first", tmp_path/"second"
    manifest = build(first, SMALL_COUNTS)
    assert manifest == build(second, SMALL_COUNTS)
    assert manifest["validation"]["records_verified"] == 90
    assert manifest["trainer_allowlist"] == ["train.json", "dev.json", "normalization.json"]
    for path, metadata in manifest["files"].items():
        assert sha256(first/path) == metadata["sha256"] == sha256(second/path)
    assert manifest["files"]["protected/final_iid.json"]["exposure"].startswith("protected")
    assert not any(group["train_ood_shared"] for group in manifest["structural_overlap"].values())
    assert all(group["train_iid_shared"] for group in manifest["structural_overlap"].values())
    assert "shares binary syntax" in manifest["structural_overlap"]["mathematics"]["ood_note"]
    assert manifest["generation_counters"]["train"]["mathematics"]["accepted"] == 12
    assert manifest["generation_counters"]["train"]["coding"]["accepted"] == 12
    with pytest.raises(FileExistsError):
        build(first, SMALL_COUNTS)


def test_normalization_uses_only_training_targets(tmp_path):
    output = tmp_path/"data"
    build(output, SMALL_COUNTS)
    train = json.loads((output/"train.json").read_text())
    norm = json.loads((output/"normalization.json").read_text())
    assert norm["computed_from"] == "train-only"
    states = [row["target"]["state"] for row in train if row["domain"] == "physics"]
    assert norm["physics_state_mean"] == [statistics.mean(axis) for axis in zip(*states)]
    assert norm["physics_state_std"] == [max(statistics.pstdev(axis), 1e-6) for axis in zip(*states)]
    assert norm == normalization(train, sha256(output/"train.json"))


def test_verify_all_domains_and_reject_target_or_feature_tampering():
    for module in (mathematics, physics, coding):
        row = module.generate("train", 1, 1)[0]
        assert verify_row(row)
        bad = copy.deepcopy(row)
        if row["domain"] == "physics":
            bad["target"]["state"][0] += 1
        else:
            bad["target"]["answer"] = "wrong"
        with pytest.raises(ValueError):
            verify_row(bad)
        bad = copy.deepcopy(row)
        bad["features"] = [999]
        with pytest.raises(ValueError):
            verify_row(bad)


def test_coding_structured_ast_tampering_rejected():
    row = coding.generate("train", 4, 1)[0]
    row["structured_input"]["ast"] = "wrong"
    with pytest.raises(ValueError):
        verify_row(row)


def test_global_duplicate_and_misassignment_detection():
    row = mathematics.generate("train", 7, 1)[0]
    with pytest.raises(ValueError, match="Duplicate"):
        validate_splits({"train": [row, row]})
    with pytest.raises(ValueError, match="wrong split"):
        validate_splits({"dev": [row]})


def test_training_normalization_requires_domains():
    with pytest.raises(ValueError):
        normalization([], "hash")


@pytest.mark.parametrize("counts", [{}, {"train": 0, "dev": 1, "final_iid": 1, "final_ood": 1}])
def test_invalid_counts_create_no_output(tmp_path, counts):
    path = tmp_path/"invalid"
    with pytest.raises(ValueError):
        build(path, counts)
    assert not path.exists()
