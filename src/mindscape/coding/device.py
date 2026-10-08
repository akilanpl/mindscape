"""Explicit neural device selection; accelerated studies never silently use CPU."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelDevice:
    device: object
    dtype: object
    name: str
    precision: str

    def synchronize(self):
        if self.name == "mps":
            import torch

            torch.mps.synchronize()

    def check_model(self, model):
        misplaced = [name for name, p in model.named_parameters() if p.device.type != self.name]
        misplaced += ["buffer:" + name for name, p in model.named_buffers() if p.device.type != self.name]
        if misplaced:
            raise RuntimeError("Neural parameters on wrong device: " + str(misplaced[:5]))


def select_device(request=None, precision=None, require_mps=False):
    import torch

    request = request or os.getenv("MINDSCAPE_DEVICE", "mps")
    precision = precision or os.getenv("MINDSCAPE_PRECISION", "float32")
    if request not in ("mps", "cpu", "auto"):
        raise ValueError("Unknown neural device")
    if request == "auto":
        request = "mps" if torch.backends.mps.is_available() else "cpu"
    if require_mps and request != "mps":
        raise RuntimeError("Accelerated neural evaluation requires MPS; CPU fallback forbidden")
    if request == "mps":
        if os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
            raise RuntimeError("Silent MPS CPU fallback is forbidden")
        if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS unavailable: built=" + str(torch.backends.mps.is_built())
                + ", available=" + str(torch.backends.mps.is_available())
                + ". Main neural evaluation stopped; no CPU fallback."
            )
    dtypes = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if precision not in dtypes:
        raise ValueError("Unsupported precision; float64 is forbidden")
    selected = ModelDevice(torch.device(request), dtypes[precision], request, precision)
    # An actual allocation catches device-access failures before loading large weights.
    torch.ones(1, device=selected.device, dtype=selected.dtype)
    selected.synchronize()
    return selected
