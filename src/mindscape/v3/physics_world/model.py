"""Action-conditioned MPS neural predictions. No authoritative simulator dependency."""

import numpy as np
import torch
from torch import nn

from .accelerator import require_mps, verify_placement
from .schema import DT, HISTORY, STATE_DIM

VARIANTS = (
    "mindscape",
    "matched",
    "no_relationships",
    "no_concepts",
    "no_memory",
    "no_history",
    "no_feedback",
)
INPUT_DIM = HISTORY * STATE_DIM * 2 + 4 + 4 + 6 + 6


def encode(history, masks, actions, context, variant):
    if history.device.type != "mps":
        raise RuntimeError("CPU neural encoding forbidden")
    if variant == "no_history":
        history = history[:, -1:].expand_as(history)
        masks = masks[:, -1:].expand_as(masks)
    current = history[:, -1]
    active = current[..., 9:10]
    base = torch.cat(
        (
            history.permute(0, 2, 1, 3).flatten(2),
            masks.permute(0, 2, 1, 3).flatten(2),
            actions,
            context[:, None, :].expand(-1, current.shape[1], -1),
        ),
        dim=-1,
    )
    relative = current[:, None, :, :4] - current[:, :, None, :4]
    valid = active[:, None, :, 0] * active[:, :, None, 0]
    diagonal = torch.eye(current.shape[1], device="mps")[None]
    valid = valid * (1 - diagonal)
    distance = relative[..., :2].square().sum(-1).sqrt()
    approach = (relative[..., :2] * relative[..., 2:4]).sum(-1) / (distance + 0.01)
    relations = torch.cat((relative, distance[..., None], approach[..., None]), dim=-1)
    relations = (relations * valid[..., None]).sum(2) / valid.sum(2).clamp(min=1)[..., None]
    velocity = (history[:, -1, :, :2] - history[:, -2, :, :2]) / DT
    acceleration = (history[:, -1, :, 2:4] - history[:, -2, :, 2:4]) / DT
    momentum = current[..., 2:4] * current[..., 4:5]
    concepts = torch.cat((velocity, acceleration, momentum), dim=-1)
    if variant in ("matched", "no_relationships"):
        relations = relations * 0
    if variant in ("matched", "no_concepts"):
        concepts = concepts * 0
    return torch.cat((base, relations, concepts), dim=-1) * active


class WorldModel(nn.Module):
    def __init__(self, variant="mindscape", seed=11):
        device = require_mps()
        if variant not in VARIANTS:
            raise ValueError("Unknown variant")
        torch.manual_seed(seed)
        self.variant = variant
        with torch.device(device):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(INPUT_DIM, 96), nn.SiLU(), nn.Linear(96, 96), nn.SiLU()
            )
            self.predictor = nn.Sequential(nn.Linear(192, 96), nn.SiLU(), nn.Linear(96, 9))
            self.register_buffer("target_scale", torch.ones(4, device=device))
            self.register_buffer("feature_scale", torch.ones(INPUT_DIM, device=device))
        self.memory_keys = None
        self.memory_residuals = None
        self.memory_active = None
        verify_placement(self)

    def forward(self, history, masks, actions, context):
        verify_placement(self, history, masks, actions, context)
        features = encode(history, masks, actions, context, self.variant) / self.feature_scale
        latent = self.encoder(features)
        active = history[:, -1, :, 9:10]
        # All variants share capacity; matched removes relational pooling and derived concepts.
        pooled = (latent * active).sum(1, keepdim=True) / active.sum(1, keepdim=True).clamp(min=1)
        if self.variant in ("matched", "no_relationships"):
            pooled = pooled * 0
        raw = self.predictor(torch.cat((latent, pooled.expand_as(latent)), dim=-1))
        return raw[..., :4], raw[..., 4:8].clamp(-6, 3), raw[..., 8], features

    def predict(self, history, masks, actions, context):
        with torch.inference_mode():
            mean, logvar, contact, features = self(history, masks, actions, context)
            if self.memory_keys is not None and self.variant not in ("matched", "no_memory"):
                distance = torch.cdist(features.flatten(1), self.memory_keys)
                values, indices = distance.topk(min(3, distance.shape[1]), largest=False)
                weights = torch.softmax(-values, dim=-1)
                residual = (self.memory_residuals[indices] * weights[..., None, None]).sum(1)
                mean = mean + 0.1 * residual
            active = history[:, -1, :, 9:10]
            return (
                mean * self.target_scale * active,
                torch.exp(logvar) * self.target_scale.square() * active,
                torch.sigmoid(contact) * active[..., 0],
            )

    def build_memory(self, arrays, approved_ids):
        if not approved_ids or arrays["trajectory_ids"] != approved_ids:
            raise ValueError("Training-approved memory required")
        if self.variant in ("matched", "no_memory"):
            return
        # Bounded deterministic training-only exemplars, indexed evenly rather than by success.
        indices = np.linspace(
            0, len(arrays["history"]) - 1, min(256, len(arrays["history"])), dtype=int
        )
        batch = {
            key: torch.as_tensor(arrays[key][indices], device="mps", dtype=torch.float32)
            for key in ("history", "masks", "actions", "context", "delta")
        }
        with torch.inference_mode():
            mean, _, _, features = self(
                batch["history"], batch["masks"], batch["actions"], batch["context"]
            )
            self.memory_keys = features.flatten(1).detach()
            self.memory_residuals = (batch["delta"] / self.target_scale - mean).detach()
        self.memory_active = {
            "split": "train",
            "trajectory_ids": approved_ids,
            "transitions": len(indices),
            "kind": "simulator-verified training transitions, not learned rule truth",
        }


def configure_scales(model, arrays):
    # Training-only normalization; no holdout targets or features enter it.
    target = arrays["delta"]
    active = arrays["history"][:, -1, :, 9] > 0
    scale = np.maximum(target[active].std(0), 0.01).astype(np.float32)
    model.target_scale.copy_(torch.as_tensor(scale, device="mps"))
    with torch.inference_mode():
        x = encode(
            *(
                torch.as_tensor(arrays[key], device="mps", dtype=torch.float32)
                for key in ("history", "masks", "actions", "context")
            ),
            model.variant,
        )
        model.feature_scale.copy_(x.flatten(0, 1).std(0).clamp(min=0.1))
