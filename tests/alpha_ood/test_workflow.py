import json
from pathlib import Path

import pytest

from mindscape.alpha_ood.benchmark import digest
from mindscape.alpha_ood.preparation import REQUIRED
from mindscape.alpha_ood.workflow import BoundaryError, execute_prepared_campaign


def test_missing_gate_never_calls_protected_callbacks():
    protocol = json.loads(Path("configs/alpha_ood/protocol-v1.json").read_text())
    calls = []
    with pytest.raises(BoundaryError):
        execute_prepared_campaign(protocol=protocol, source_hash="a" * 64,
                                  input_hash="b" * 64, label_commitment="c" * 64,
                                  receipts={}, verify_receipt=lambda *args: True,
                                  fit=lambda x: calls.append(x), predict=lambda *args: [],
                                  persist_seal=lambda x: calls.append(x))
    assert calls == []


def test_synthetic_signed_bindings_serialize_and_preserve_failure():
    protocol = json.loads(Path("configs/alpha_ood/protocol-v1.json").read_text())
    binding = {"protocol_hash": digest(protocol), "source_hash": "a" * 64,
               "input_hash": "b" * 64, "label_commitment": "c" * 64}
    receipts = {k: {"binding": binding, "approved": True, "synthetic_fixture": True} for k in REQUIRED}
    calls = []
    saved = []

    def fake_fit(spec):  # Fixture: no model, optimizer, data or scientific prediction.
        calls.append(spec)
        if len(calls) == 2:
            raise RuntimeError("synthetic failure")
        return "synthetic checkpoint"

    report = execute_prepared_campaign(
        protocol=protocol, source_hash="a" * 64, input_hash="b" * 64,
        label_commitment="c" * 64, receipts=receipts,
        verify_receipt=lambda key, receipt: receipt["synthetic_fixture"],
        fit=fake_fit, predict=lambda *args: [{"synthetic": True}], persist_seal=saved.append)
    assert report["scientific_status"] == "FAILED_CAMPAIGN"
    assert len(report["completed"]) == 1
    assert len(calls) == 2
    assert len(report["predictions"]) == 1
    assert saved == [report]
