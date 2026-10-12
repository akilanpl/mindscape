"""Reproduce neural evaluator weights from frozen executed training labels, not new timings."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from mindscape.v3.coding_learning.planning import fit

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--training", required=True)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    root = Path(a.training)
    ledger = json.loads((root / "ledger.json").read_text())
    evidence = json.loads((root / "training-evidence.json").read_text())
    original = root / f"evaluator-s{a.seed}.pt"
    expected = next(r["checkpoint_sha256"] for r in ledger["receipts"] if r["seed"] == a.seed)
    if hashlib.sha256(original.read_bytes()).hexdigest() != expected:
        raise ValueError("Original checkpoint changed")
    started = time.perf_counter()
    model, receipt = fit(evidence["rows"], a.seed)
    reference = torch.load(original, weights_only=True)["state"]
    receipt.update(
        optimizer_loop_seconds=time.perf_counter() - started,
        scope="reproduction from frozen training evidence, not original training timing",
        every_tensor_identical=all(
            torch.equal(value, reference[name]) for name, value in model.state_dict().items()
        ),
    )
    Path(a.output).write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))
    if not receipt["every_tensor_identical"]:
        raise ValueError("Deterministic training reproduction failed")
