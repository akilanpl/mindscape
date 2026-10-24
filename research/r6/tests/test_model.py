"""CPU inference/serialization only: no optimizer, backward or local fitting."""
import ast
import copy
from pathlib import Path
import pytest
import torch

from research.r6 import common
from research.r6.architectures.model import make_model, predict
from research.r6.datasets import mathematics, physics, coding


@pytest.fixture
def rows():
    return [module.generate("train", 61017, 1)[0] for module in (mathematics, physics, coding)]


@pytest.mark.parametrize("text", ["", "a" * (common.MAX_INPUT + 1)])
def test_reject_input_outside_declared_length(rows, text):
    rows[0]["input"] = text
    with pytest.raises(ValueError):
        common.encode_batch(rows, "cpu")


def test_ascii_input_and_empty_batch_rejected(rows):
    rows[0]["input"] = "λ"
    with pytest.raises(UnicodeEncodeError):
        common.encode_batch(rows, "cpu")
    with pytest.raises(ValueError):
        common.encode_batch([], "cpu")


@pytest.mark.parametrize("answer", ["01", "-0", "+1", "1.0", "1 ", "", "--1", 1, "9" * common.MAX_OUTPUT])
def test_noncanonical_and_out_of_range_targets_rejected(answer):
    with pytest.raises(ValueError):
        common.encode_answer(answer)


@pytest.mark.parametrize("architecture,width", [("native-relational", 48), ("recurrent", 58)])
def test_cpu_forward_shapes_and_domain_masks(rows, architecture, width):
    model = make_model({"architecture": architecture, "width": width}).eval()
    batch = common.encode_batch(rows, "cpu")
    with torch.inference_mode():
        output = model(batch["tokens"], batch["lengths"], batch["features"], batch["domains"], batch["target_input"])
    assert output["text_logits"].shape == (*batch["target_input"].shape, common.VOCAB)
    assert output["physics"].shape == (3, 4)
    assert batch["physics_mask"].tolist() == [False, True, False]
    assert batch["text_mask"].tolist() == [True, False, True]
    assert torch.isfinite(output["text_logits"]).all()
    assert torch.isfinite(output["physics"]).all()
    assert all(value.device.type == "cpu" for value in model.state_dict().values())


def test_numeric_property_path_masked_outside_physics(rows):
    model = make_model({"width": 48}).eval()
    batch = common.encode_batch(rows, "cpu")
    altered = batch["features"].clone()
    altered[0] = 99
    altered[2] = -99
    with torch.inference_mode():
        first = model.encode(batch["tokens"], batch["lengths"], batch["features"], batch["domains"])
        second = model.encode(batch["tokens"], batch["lengths"], altered, batch["domains"])
    assert torch.equal(first, second)


def test_predictions_respond_to_observed_physics_and_ignore_labels(rows):
    model = make_model({"width": 48}).eval()
    original = predict(model, rows)
    changed = copy.deepcopy(rows)
    changed[1]["features"][5] += 10
    altered = predict(model, changed)
    assert original[1]["state"] != altered[1]["state"]
    # Inference has no target-answer dependence, despite using the common codec.
    relabeled = copy.deepcopy(rows)
    relabeled[0]["target"] = {"answer": "0"}
    relabeled[1]["target"] = {"state": [1000., 1000., 1000., 1000.]}
    relabeled[2]["target"] = {"answer": "0"}
    assert predict(model, relabeled) == original
    assert len(original) == 3
    assert set(original[0]) == {"answer", "raw_text"}
    assert set(original[1]) == {"state"}
    assert set(original[2]) == {"answer", "raw_text"}


def test_sequence_context_depends_on_observed_text(rows):
    model = make_model({"width": 48}).eval()
    changed = copy.deepcopy(rows)
    changed[0]["input"] = "Calculate (99) + (99)."
    first, second = common.encode_batch(rows, "cpu"), common.encode_batch(changed, "cpu")
    with torch.inference_mode():
        one = model.encode(first["tokens"], first["lengths"], first["features"], first["domains"])
        two = model.encode(second["tokens"], second["lengths"], second["features"], second["domains"])
    assert not torch.equal(one[0], two[0])
    # This is dependency of an initialized network, never learned arithmetic accuracy.


def test_safe_json_checkpoint_exact_reload_and_metadata(tmp_path, rows):
    config = {"width": 48, "physics_target_mean": [1., 2., 3., 4.], "physics_target_scale": [2., 3., 4., 5.]}
    model, restored = make_model(config), make_model({**config, "seed": 9})
    path = tmp_path / "checkpoint.json"
    metadata = {"test": "inference-only", "optimizer_updates": 0}
    digest = common.save_checkpoint(path, model, metadata)
    assert digest == common.sha(path)
    assert common.load_checkpoint(path, restored) == metadata
    assert common.read(path)["scientifically_qualified"] is False
    assert all(torch.equal(value, restored.state_dict()[name]) for name, value in model.state_dict().items())
    assert predict(model, rows) == predict(restored, rows)
    with pytest.raises(FileExistsError):
        common.save_checkpoint(path, model, metadata)


def test_checkpoint_rejects_unknown_schema_and_nonfinite_tensors(tmp_path):
    model = make_model({"width": 48})
    path = tmp_path / "checkpoint.json"
    common.save_checkpoint(path, model, {})
    payload = common.read(path)
    payload["tensors"].pop(next(iter(payload["tensors"])))
    common.write(tmp_path / "missing.json", payload)
    with pytest.raises(ValueError):
        common.load_checkpoint(tmp_path / "missing.json", model)
    with torch.no_grad():
        next(model.parameters()).view(-1)[0] = float("nan")
    with pytest.raises(ValueError):
        common.save_checkpoint(tmp_path / "nonfinite.json", model, {})


def test_model_common_have_no_oracle_or_training_imports():
    root = Path(common.__file__).parent
    imported = []
    for path in (root / "common.py", root / "architectures/model.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
    assert not any(any(part in module.split(".") for part in ("datasets", "training", "evaluation")) for module in imported)
    assert not any("sympy" in module or "subprocess" in module for module in imported)


def test_parameter_matched_recurrent_candidate_without_fitting():
    count = lambda model: sum(parameter.numel() for parameter in model.parameters())
    native = count(make_model({"architecture": "native-relational", "width": 48}))
    candidates = [(abs(count(make_model({"architecture": "recurrent", "width": width})) - native), width)
                  for width in range(24, 100)]
    _, width = min(candidates)
    recurrent = count(make_model({"architecture": "recurrent", "width": width}))
    assert width == 58
    assert native == 54014
    assert recurrent == 53776
    assert abs(recurrent - native) / native < .05
