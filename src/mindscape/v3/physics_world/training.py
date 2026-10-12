"""MPS-only fixed-update fits, optimizer-state checkpoint continuation and audit."""

import json
import time
from pathlib import Path

import numpy as np
import torch

from .accelerator import require_mps, verify_placement
from .data import training_arrays
from .model import WorldModel, configure_scales
from .storage import atomic_json, file_sha


def checkpoint(path, model, optimizer, step, identity):
    state = {
        "state": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "identity": identity,
        "memory_keys": model.memory_keys,
        "memory_residuals": model.memory_residuals,
        "memory_active": model.memory_active,
    }

    # Device-to-CPU serialization is allowed; no CPU neural computation.
    def cpu(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {k: cpu(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cpu(v) for v in value]
        return value

    generation = Path(path).parent / f"checkpoint-step{step:06d}.pt"
    if generation.exists():
        generation = Path(path).parent / f"checkpoint-step{step:06d}-{time.time_ns()}.pt"
    tmp = generation.with_suffix(".tmp")
    torch.save(cpu(state), tmp)
    tmp.replace(generation)
    atomic_json(
        path,
        {
            "file": generation.name,
            "sha256": file_sha(generation),
            "step": step,
            "identity": identity,
        },
    )


def restore(path, model, optimizer, identity):
    receipt = json.loads(Path(path).read_text())
    generation = Path(path).parent / receipt["file"]
    if (
        Path(receipt["file"]).name != receipt["file"]
        or file_sha(generation) != receipt["sha256"]
        or receipt["identity"] != identity
    ):
        raise ValueError("Checkpoint hash/identity changed")
    saved = torch.load(generation, map_location="mps", weights_only=True)
    if saved["identity"] != identity:
        raise ValueError("Checkpoint provenance")
    model.load_state_dict(saved["state"])
    optimizer.load_state_dict(saved["optimizer"])
    model.memory_keys = saved["memory_keys"]
    model.memory_residuals = saved["memory_residuals"]
    model.memory_active = saved["memory_active"]
    verify_placement(model)
    return saved["step"]


def train(dataset, output, identity, updates=400, batch_size=64, stop_after=None):
    require_mps()
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    arrays = training_arrays(dataset, identity["family"], identity["budget"])
    model = WorldModel(identity["variant"], identity["seed"])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    path = root / "checkpoint-index.json"
    step = 0
    if path.exists():
        step = restore(path, model, optimizer, identity)
    else:
        configure_scales(model, arrays)
    tensors = {
        k: torch.as_tensor(arrays[k], device="mps", dtype=torch.float32)
        for k in ("history", "masks", "actions", "context", "delta", "contact")
    }
    rows = len(arrays["history"])
    active = tensors["history"][:, -1, :, 9]
    feedback_targets = tensors["delta"] / model.target_scale
    if identity["variant"] == "no_feedback":
        order = np.random.default_rng(identity["seed"] + 871).permutation(rows)
        feedback_targets = feedback_targets[torch.as_tensor(order, device="mps")]
    torch.mps.synchronize()
    started = time.perf_counter()
    history = []
    initial_step = step
    end = min(updates, stop_after) if stop_after is not None else updates
    for update in range(step, end):
        indices = np.random.default_rng(identity["seed"] * 1000003 + update).integers(
            0, rows, size=min(batch_size, rows)
        )
        index = torch.as_tensor(indices, device="mps")
        optimizer.zero_grad(set_to_none=True)
        mean, logvar, contact, _ = model(
            *(tensors[k][index] for k in ("history", "masks", "actions", "context"))
        )
        error = (mean - feedback_targets[index]).square()
        nll = 0.5 * (error * torch.exp(-logvar) + logvar)
        valid = active[index]
        loss = (nll * valid[..., None]).sum() / valid.sum().clamp(min=1) / 4
        loss = loss + 0.1 * (
            torch.nn.functional.binary_cross_entropy_with_logits(
                contact, tensors["contact"][index], reduction="none"
            )
            * valid
        ).sum() / valid.sum().clamp(min=1)
        loss.backward()
        if update == initial_step:
            verify_placement(
                model,
                *(tensors[k][index] for k in ("history", "masks", "actions", "context")),
                gradients=True,
            )
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        step = update + 1
        if step % 50 == 0 or step == end:
            torch.mps.synchronize()
            value = float(loss.detach().item())
            if not np.isfinite(value):
                raise RuntimeError("Nonfinite GPU training loss")
            history.append({"update": step, "loss": value})
            checkpoint(path, model, optimizer, step, identity)
            atomic_json(
                root / "progress.json",
                {
                    "step": step,
                    "required_updates": updates,
                    "identity": identity,
                    "device": "mps",
                    "loss": value,
                },
            )
    if step == updates and initial_step < updates:
        model.build_memory(arrays, arrays["trajectory_ids"])
        checkpoint(path, model, optimizer, step, identity)
    torch.mps.synchronize()
    elapsed = time.perf_counter() - started
    receipt = {
        "identity": identity,
        "step": step,
        "complete": step == updates,
        "new_updates": step - initial_step,
        "batch_size": batch_size,
        "distinct_training_trajectories": len(arrays["trajectory_ids"]),
        "transition_labels": rows,
        "new_transition_presentations": (step - initial_step) * min(batch_size, rows),
        "training_trajectory_ids": arrays["trajectory_ids"],
        "device": "mps",
        "precision": "float32",
        "loss_history": history,
        "seconds": elapsed,
        "updates_per_second": (step - initial_step) / elapsed if elapsed else None,
        "current_allocated_bytes": torch.mps.current_allocated_memory(),
        "driver_allocated_bytes": torch.mps.driver_allocated_memory(),
        "checkpoint_sha256": file_sha(path),
        "placement_verified": True,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    atomic_json(root / f"training-invocation-{time.time_ns()}.json", receipt)
    return model, receipt
