import json
from pathlib import Path

import pytest
import torch

from mindscape.alpha_ood.models import LocalTransition
from mindscape.alpha_ood.preparation import fit_plan, readiness


def test_matching_initialization_common_heads_and_approximate_capacity():
    recurrent = LocalTransition("recurrent", 11)
    feedforward = LocalTransition("feedforward", 11)
    repeat = LocalTransition("recurrent", 11)
    assert all(torch.equal(a, b) for a, b in zip(recurrent.parameters(), repeat.parameters()))
    # Common layers are initialized before differing core shapes consume RNG draws.
    assert torch.equal(recurrent.operation.weight, feedforward.operation.weight)
    assert torch.equal(recurrent.digit_head.weight, feedforward.digit_head.weight)
    assert torch.equal(recurrent.sign_head[0].weight, feedforward.sign_head[0].weight)
    assert recurrent.parameter_receipt()["core_parameters"] == 15840
    assert feedforward.parameter_receipt()["core_parameters"] == 15744
    assert abs(15840 - 15744) / 15840 < .01


def test_synthetic_hidden_state_behavior_and_active_gradients():
    indices = torch.tensor([[0, 2, 3, 0]])
    relations = torch.zeros((1, 7))
    zero = torch.zeros((1, 48))
    nonzero = torch.ones((1, 48))
    ff = LocalTransition("feedforward", 23)
    recurrent = LocalTransition("recurrent", 23)
    assert torch.equal(ff(indices, relations, zero)[0], ff(indices, relations, nonzero)[0])
    assert not torch.equal(recurrent(indices, relations, zero)[0],
                           recurrent(indices, relations, nonzero)[0])
    d, c, _ = recurrent(indices, relations, nonzero)
    (d.sum() + c.sum()).backward()  # Synthetic gradient check; zero optimizer updates.
    assert recurrent.core.weight_hh.grad.abs().sum() > 0


def test_preparation_cannot_self_authorize():
    protocol = json.loads(Path("configs/alpha_ood/protocol-v1.json").read_text())
    assert len(fit_plan(protocol)) == 6
    receipt = readiness(protocol, {})
    assert receipt["scientific_status"] == "UNRUN"
    assert not receipt["execution_permitted"]
    assert len(receipt["missing_receipts"]) == 8
    with pytest.raises(ValueError):
        LocalTransition("procedural", 11)
