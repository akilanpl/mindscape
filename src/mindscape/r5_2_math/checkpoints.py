"""CPU tensor inspection only; no neural forwards, no historical checkpoint mutation."""

import hashlib

import torch


def audit(root, fits, protocol):
    if len(fits) != 6 or len({(f["kind"], f["seed"]) for f in fits}) != 6:
        raise ValueError("Fit coverage")
    if len({f["parameters"] for f in fits}) != 1:
        raise ValueError("Unmatched neural parameter budgets")
    for f in fits:
        path = root / f["checkpoint"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != f["sha256"]:
            raise ValueError("Checkpoint hash")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if (
            saved["updates"] != protocol["updates_per_fit"]
            or saved["losses"] != f["losses"]
            or saved["seed"] != f["seed"]
            or saved["kind"] != f["kind"]
        ):
            raise ValueError("Checkpoint receipt")
        if len(saved["losses"]) != 800 or not all(
            torch.isfinite(v).all() for v in saved["state"].values()
        ):
            raise ValueError("Nonfinite checkpoint")
        if sum(v.numel() for v in saved["state"].values()) != f["parameters"]:
            raise ValueError("Parameter count")
        if not saved["optimizer"]["state"] or any(
            v["step"].item() != 800
            or not torch.isfinite(v["exp_avg"]).all()
            or not torch.isfinite(v["exp_avg_sq"]).all()
            for v in saved["optimizer"]["state"].values()
        ):
            raise ValueError("Optimizer state")
    return {"status": "PASS", "checkpoints": 6, "neural_re_forward": False}
