"""Read-only runtime probe; writes a new receipt, never changes environment/security."""

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import torch

from mindscape.v3.physics_world.accelerator import require_mps, verify_placement


def diagnose(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"native-diagnostic-{time.time_ns()}.json"
    result = {
        "schema": "mindscape.v3.physics.native-probe/1",
        "interpreter": sys.executable,
        "python": platform.python_version(),
        "architecture": platform.machine(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "environment_indicators": {
            k: os.environ.get(k)
            for k in (
                "CODEX_SANDBOX",
                "CODEX_CI",
                "CODEX_PERMISSION_PROFILE",
                "PYTORCH_ENABLE_MPS_FALLBACK",
            )
        },
        "scope": "Diagnostic neural computation only; no research dataset/model download/environment modification",
    }
    try:
        device = require_mps()
        with torch.device(device):
            model = torch.nn.Linear(4, 2)
            model.register_buffer("probe", torch.zeros(1, device=device))
            x = torch.ones(16, 4, device=device)
            y = torch.zeros(16, 2, device=device)
        verify_placement(model, x, y)
        before = model.weight.detach().clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss = (model(x) - y).square().mean()
        loss.backward()
        verify_placement(model, x, y, gradients=True)
        optimizer.step()
        torch.mps.synchronize()
        if torch.equal(before, model.weight.detach()):
            raise RuntimeError("Optimizer did not update")
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("Nonfinite loss")
        result.update(
            status="PASS",
            allocation=True,
            forward_backward=True,
            optimizer_update=True,
            placement_verified=True,
            device="mps",
            precision="float32",
        )
    except (RuntimeError, TypeError, ValueError) as error:
        result.update(status="BLOCKED", error=str(error))
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps({"receipt": str(path.resolve()), **result}, indent=2))
    return result["status"] == "PASS"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", default="results/v3_physics/native-diagnostics")
    sys.exit(0 if diagnose(p.parse_args().output_root) else 2)
