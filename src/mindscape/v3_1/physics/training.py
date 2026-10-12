"""Actual free-running multi-step supervision, MPS only; safe resumable state."""

import time
from pathlib import Path

import numpy as np
import torch

from mindscape.v3.physics_world.accelerator import require_mps, verify_placement
from mindscape.v3.physics_world.storage import atomic_json, file_sha
from mindscape.v3.physics_world.training import checkpoint, restore

from .data import windows
from .model import Dynamics


def sequence(model, history, masks, actions, contexts):
    states = []
    variances = []
    contacts = []
    for t in range(actions.shape[1]):
        state, var, contact, _ = model(history, masks, actions[:, t], contexts[:, t])
        states.append(state)
        variances.append(var)
        contacts.append(contact)
        current = torch.cat(
            (state, history[:, -1, :, 4:8], contact[..., None], history[:, -1, :, 9:10]), -1
        )
        history = torch.cat((history[:, 1:], current[:, None]), 1)
        status = masks[:, -1].clone()
        status[..., :4] = 2 * current[..., 9:10]  # 2 means predicted, not observed.
        masks = torch.cat((masks[:, 1:], status[:, None]), 1)
    return torch.stack(states, 1), torch.stack(variances, 1), torch.stack(contacts, 1)


def fit(dataset, root, identity, protocol, stop_after=None):
    require_mps()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    arrays = windows(dataset, identity["family"], protocol["training_rollout"])
    model = Dynamics(identity["variant"], identity["seed"])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-5)
    pointer = root / "checkpoint-index.json"
    step = restore(pointer, model, optimizer, identity) if pointer.exists() else 0
    tensors = {
        k: torch.as_tensor(v, device="mps", dtype=torch.float32)
        for k, v in arrays.items()
        if k != "ids"
    }
    active = tensors["history"][:, -1, :, 9]
    started = time.perf_counter()
    initial = step
    end = min(protocol["updates"], stop_after) if stop_after is not None else protocol["updates"]
    for update in range(step, end):
        indices = np.random.default_rng(identity["seed"] * 100003 + update).integers(
            len(active), size=protocol["batch"]
        )
        idx = torch.as_tensor(indices, device="mps")
        optimizer.zero_grad(set_to_none=True)
        states, var, contact = sequence(
            model,
            tensors["history"][idx],
            tensors["masks"][idx],
            tensors["actions"][idx],
            tensors["contexts"][idx],
        )
        error = states - tensors["targets"][idx]
        valid = active[idx, None, :, None]
        scale = torch.tensor([1, 1, 3, 3], device="mps")
        mse = ((error / scale).square() * valid).sum() / valid.sum() / states.shape[1] / 4
        cumulative = var.cumsum(1).clamp(min=1e-6)
        nll = (
            ((error.square() / cumulative + cumulative.log()) * valid).sum()
            / valid.sum()
            / states.shape[1]
            / 8
        )
        bce = torch.nn.functional.binary_cross_entropy(
            contact.clamp(1e-6, 1 - 1e-6), tensors["contacts"][idx], reduction="none"
        )
        loss = mse + 0.02 * nll + 0.02 * (bce * active[idx, None]).mean()
        loss.backward()
        if update == initial:
            verify_placement(model, tensors["history"][idx], gradients=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        step = update + 1
        if step % 40 == 0 or step == end:
            torch.mps.synchronize()
            value = float(loss.item())
            if not np.isfinite(value):
                raise RuntimeError("Nonfinite training loss")
            checkpoint(pointer, model, optimizer, step, identity)
            atomic_json(root / "progress.json", {"step": step, "loss": value, "device": "mps"})
    torch.mps.synchronize()
    receipt = {
        "identity": identity,
        "step": step,
        "new_updates": step - initial,
        "device": "mps",
        "precision": "float32",
        "placement_verified": True,
        "seconds": time.perf_counter() - started,
        "trajectory_ids": arrays["ids"],
        "training_windows": len(active),
        "window_presentations": (step - initial) * protocol["batch"],
        "intermediate_labels_per_window": protocol["training_rollout"],
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "checkpoint_index_hash": file_sha(pointer),
    }
    atomic_json(root / f"training-{time.time_ns()}.json", receipt)
    return model, {**tensors, "ids": arrays["ids"]}
