"""Hardware gate and conditional MPS microbenchmarks; write receipts even on failure."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

from mindscape.v3.physics_world.accelerator import require_mps, verify_placement


def hardware():
    receipt = {
        "python": sys.version,
        "interpreter": sys.executable,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
        "device_required": "mps",
        "neural_cpu_fallback_permitted": False,
    }
    result = subprocess.run(
        ["system_profiler", "SPHardwareDataType", "SPDisplaysDataType", "-json"],
        capture_output=True,
        text=True,
        check=False,
    )
    receipt["hardware_probe_returncode"] = result.returncode
    if result.returncode == 0:
        info = json.loads(result.stdout)
        receipt["hardware"] = [
            {
                key: row[key]
                for key in ("chip_type", "machine_model", "physical_memory")
                if key in row
            }
            for row in info.get("SPHardwareDataType", [])
        ]
        receipt["metal"] = [
            row.get("spdisplays_metal") for row in info.get("SPDisplaysDataType", [])
        ]
    else:
        receipt["hardware_probe_error"] = result.stderr[:2000]
    return receipt


def benchmark(device, precision, batch):
    dtype = getattr(torch, precision)
    torch.manual_seed(11)
    # Synthetic throughput fixture, not physics data, scientific training or a checkpoint.
    with torch.device(device):
        model = torch.nn.Sequential(
            torch.nn.Linear(40, 128), torch.nn.Tanh(), torch.nn.Linear(128, 12)
        ).to(dtype=dtype)
        model.register_buffer("audit_buffer", torch.zeros(1, device=device, dtype=dtype))
        x = torch.randn(batch, 40, device=device, dtype=dtype)
        y = torch.randn(batch, 12, device=device, dtype=dtype)
    verify_placement(model, x, y)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    initial = next(model.parameters()).detach().clone()
    losses = []
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        loss = (model(x) - y).square().mean()
        loss.backward()
        verify_placement(model, x, y, gradients=True)
        optimizer.step()
    torch.mps.synchronize()
    start = time.perf_counter()
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        loss = (model(x) - y).square().mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.detach())
    torch.mps.synchronize()
    training_seconds = time.perf_counter() - start
    if not all(bool(torch.isfinite(loss).item()) for loss in losses):
        raise RuntimeError("Nonfinite training loss")
    if torch.equal(initial, next(model.parameters()).detach()):
        raise RuntimeError("Optimizer did not update parameters")
    verify_placement(model, x, y, gradients=True)
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor) and key != "step" and value.device.type != "mps":
                raise RuntimeError("Optimizer moments misplaced")
    with torch.inference_mode():
        expected = model(x)
        torch.mps.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            observed = model(x)
        torch.mps.synchronize()
        inference_seconds = time.perf_counter() - start
        repeat_difference = float((observed - expected).abs().max().item())
    return {
        "precision": precision,
        "batch_size": batch,
        "device": "mps",
        "training_updates": 25,
        "timed_updates": 20,
        "training_seconds": training_seconds,
        "training_samples_per_second": 20 * batch / training_seconds,
        "inference_seconds": inference_seconds,
        "inference_samples_per_second": 100 * batch / inference_seconds,
        "repeat_max_difference": repeat_difference,
        "loss_start": float(losses[0].item()),
        "loss_end": float(losses[-1].item()),
        "current_allocated_bytes": torch.mps.current_allocated_memory(),
        "driver_allocated_bytes": torch.mps.driver_allocated_memory(),
        "placement_and_optimizer_verified": True,
        "scope": "Synthetic MLP feasibility; not world-model throughput or accuracy",
    }


def audit(output):
    output = Path(output)
    if output.exists():
        raise ValueError("Audit receipt exists; use a new path")
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt = hardware()
    receipt["benchmarks"] = []
    start = time.perf_counter()
    try:
        device = require_mps()
        receipt["allocation"] = "passed"
        for precision in ("float32", "float16", "bfloat16"):
            for batch in (16, 32, 64, 128):
                try:
                    receipt["benchmarks"].append(benchmark(device, precision, batch))
                except (RuntimeError, TypeError) as error:
                    receipt["benchmarks"].append(
                        {
                            "precision": precision,
                            "batch_size": batch,
                            "status": "unsupported_or_failed",
                            "error": str(error),
                        }
                    )
        # Reduced precision is exploratory until validated against physics trajectories.
        stable = [
            r
            for r in receipt["benchmarks"]
            if r.get("placement_and_optimizer_verified")
            and r["precision"] == "float32"
            and r["loss_end"] < r["loss_start"]
            and r["repeat_max_difference"] == 0
        ]
        if not stable:
            raise RuntimeError("No stable float32 benchmark configuration")
        best = max(stable, key=lambda r: r["training_samples_per_second"])
        receipt.update(
            status="hardware_gate_passed_only",
            selected_precision=best["precision"],
            selected_batch_size=best["batch_size"],
            physics_pilot_required=True,
        )
    except (RuntimeError, TypeError, ValueError) as error:
        receipt.update(
            status="BLOCKED", blocker=str(error), selected_precision=None, selected_batch_size=None
        )
        # Independent allocation diagnosis even when availability rejects the gate.
        try:
            torch.ones(1, device="mps")
            receipt["allocation_probe"] = "unexpectedly_succeeded"
        except (RuntimeError, TypeError) as allocation_error:
            receipt["allocation_probe_error"] = str(allocation_error)
    receipt.update(
        elapsed_seconds=time.perf_counter() - start,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scientific_training_seeds_completed=0,
        primary_rollouts_completed=0,
        primary_rollouts_required=6000,
        real_world_validation="NOT RUN",
    )
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt, indent=2))
    temporary.replace(output)
    print(json.dumps(receipt, indent=2))
    return receipt["status"] != "BLOCKED"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    sys.exit(0 if audit(parser.parse_args().output) else 2)
