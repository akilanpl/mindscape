"""Fail-closed MPS selection and placement verification; no CPU neural fallback."""

import os


def require_mps():
    import torch

    fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    if fallback != "0":
        raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK must be unset or0; fallback forbidden")
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError(
            f"Mandatory MPS unavailable: built={torch.backends.mps.is_built()}, available={torch.backends.mps.is_available()}; neural execution blocked"
        )
    device = torch.device("mps")
    torch.ones(1, device=device, dtype=torch.float32)
    torch.mps.synchronize()
    return device


def verify_placement(model, *inputs, gradients=False):
    tensors = list(model.named_parameters()) + list(model.named_buffers())
    tensors += [(f"input:{i}", value) for i, value in enumerate(inputs)]
    for name, value in tensors:
        if value.device.type != "mps":
            raise RuntimeError(f"Wrong neural device: {name}={value.device}")
    if gradients:
        for name, value in model.named_parameters():
            if value.requires_grad and (value.grad is None or value.grad.device.type != "mps"):
                raise RuntimeError(f"Missing or misplaced gradient: {name}")
