"""Hypothetical MPS rollouts and bounded planning, no engine answer queries."""

import numpy as np
import torch

from mindscape.v3.physics_world.accelerator import require_mps
from mindscape.v3.physics_world.planning import CANDIDATES

from .training import sequence


def rollout(model, public):
    require_mps()
    if set(public) != {"history", "masks", "actions", "contexts"}:
        raise ValueError("Private prediction field")
    tensors = {k: torch.as_tensor(v, device="mps", dtype=torch.float32) for k, v in public.items()}
    torch.mps.synchronize()
    with torch.inference_mode():
        state, var, contact = sequence(
            model, tensors["history"], tensors["masks"], tensors["actions"], tensors["contexts"]
        )
    torch.mps.synchronize()
    return {
        "states": state.cpu().numpy(),
        "variance": var.cpu().numpy(),
        "contacts": contact.cpu().numpy(),
    }


def propose(model, row, target, horizon=24):
    actions = np.zeros((9, horizon, 3, 4), np.float32)
    actions[:, 0, 0, 2:] = CANDIDATES
    contexts = np.tile(row["contexts"][3], (9, horizon, 1))
    contexts[:, :, 3] += np.arange(horizon) / 30
    public = {
        "history": np.repeat(row["observations"][None, :4], 9, 0),
        "masks": np.repeat(row["masks"][None, :4], 9, 0),
        "actions": actions,
        "contexts": contexts,
    }
    prediction = rollout(model, public)
    scores = ((prediction["states"][:, -1, 0, :2] - target) ** 2).sum(1)
    scores += 0.05 * prediction["variance"][:, -1, 0, :2].sum(1)
    scores += 2 * np.maximum(0, 0.25 - prediction["states"][:, :, 0, 1]).mean(1)
    selected = int(scores.argmin())
    return {
        "action": actions[selected, 0].tolist(),
        "scores": scores.tolist(),
        "selected": selected,
        "environment_queries": 0,
        "model_calls": 24,
        "candidate_rollouts": 9,
        "prediction": prediction["states"][selected].tolist(),
    }
