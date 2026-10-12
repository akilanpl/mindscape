"""MPS-only pairwise predictor. No physics-engine imports or private state inputs."""

import torch
from torch import nn

from mindscape.v3.physics_world.accelerator import require_mps, verify_placement
from mindscape.v3.physics_world.schema import DT

FEATURES = 54
VARIANTS = ("candidate", "matched", "no_history", "concepts", "memory")


class Dynamics(nn.Module):
    def __init__(self, variant, seed):
        device = require_mps()
        if variant not in VARIANTS:
            raise ValueError("Unknown variant")
        torch.manual_seed(seed)
        with torch.device(device):
            super().__init__()
            self.node = nn.Sequential(
                nn.Linear(FEATURES, 48), nn.SiLU(), nn.Linear(48, 48), nn.SiLU()
            )
            self.edge = nn.Sequential(
                nn.Linear(2 * FEATURES, 48), nn.SiLU(), nn.Linear(48, 48), nn.SiLU()
            )
            self.head = nn.Sequential(nn.Linear(96, 48), nn.SiLU(), nn.Linear(48, 5))
            self.register_buffer(
                "scale",
                torch.tensor(
                    ([5, 5, 6, 6, 3, 1, 1, 1, 1, 1] * 2)
                    + ([2] * 20)
                    + [5] * 4
                    + [15, 1 / 30, 10, 2]
                    + [8] * 6,
                    dtype=torch.float32,
                    device=device,
                ),
            )
            self.register_buffer("temperature", torch.ones(1, device=device))
        self.variant = variant
        self.memory_keys = None
        self.memory_residuals = None
        self.memory_active = None

    def features(self, history, masks, actions, context):
        current = history[:, -1] * (masks[:, -1] > 0)
        previous = history[:, -2] * (masks[:, -2] > 0) if self.variant != "no_history" else current
        pmask = masks[:, -2] if self.variant != "no_history" else masks[:, -1]
        concepts = torch.zeros_like(current[..., :6])
        if self.variant == "concepts":
            # Bounded, observed-only derived features; never hidden mass or future labels.
            v = (current[..., :2] - previous[..., :2]) / DT
            a = (current[..., 2:4] - previous[..., 2:4]) / DT
            concepts = torch.cat((v, a, current[..., 2:4] * current[..., 4:5]), -1).clamp(-8, 8)
        f = torch.cat(
            (
                current,
                previous,
                masks[:, -1],
                pmask,
                actions,
                context[:, None].expand(-1, current.shape[1], -1),
                concepts,
            ),
            -1,
        )
        return torch.tanh(f / self.scale)

    def forward(self, history, masks, actions, context):
        verify_placement(self, history, masks, actions, context)
        f = self.features(history, masks, actions, context)
        active = history[:, -1, :, 9]
        n = f.shape[1]
        own = f[:, :, None].expand(-1, -1, n, -1)
        neighbor = f[:, None].expand(-1, n, -1, -1)
        if self.variant != "matched":
            # Invertible relative coordinates given own state: same information, shared pair map.
            neighbor = torch.cat((neighbor[..., :4] - own[..., :4], neighbor[..., 4:]), -1)
        pairs = self.edge(torch.cat((own, neighbor), -1))
        valid = active[:, :, None] * active[:, None, :] * (1 - torch.eye(n, device=f.device)[None])
        message = (pairs * valid[..., None]).sum(2) / valid.sum(2).clamp(min=1)[..., None]
        raw = self.head(torch.cat((self.node(f), message), -1))
        dv = 6 * torch.tanh(raw[..., :2])
        if self.variant == "memory" and self.memory_keys is not None:
            distance = torch.cdist(f.flatten(0, 1), self.memory_keys)
            values, idx = distance.topk(min(3, len(self.memory_keys)), largest=False)
            correction = (torch.softmax(-values, -1)[..., None] * self.memory_residuals[idx]).sum(1)
            dv = (dv + 0.1 * correction.reshape_as(dv)).clamp(-6, 6)
        current = history[:, -1] * (masks[:, -1] > 0)
        velocity = current[..., 2:4] + dv * active[..., None]
        position = current[..., :2] + DT * velocity
        next_state = torch.cat((position, velocity), -1) * active[..., None]
        variance = torch.exp(raw[..., 2:4].clamp(-6, 4)) * self.temperature
        variance = torch.cat((variance * DT**2, variance), -1) * active[..., None]
        contact = torch.sigmoid(raw[..., 4]) * active
        return next_state, variance, contact, f

    def predict(self, history, masks, actions, context):
        with torch.inference_mode():
            state, var, contact, _ = self(history, masks, actions, context)
            return state - history[:, -1, :, :4], var, contact

    def memory(self, arrays):
        if not arrays["ids"]:
            raise ValueError("Approved training IDs required")
        index = torch.linspace(
            0, len(arrays["history"]) - 1, min(128, len(arrays["history"])), device="mps"
        ).long()
        with torch.inference_mode():
            history = arrays["history"][index]
            masks = arrays["masks"][index]
            state, _, _, features = self(
                history, masks, arrays["actions"][index, 0], arrays["contexts"][index, 0]
            )
            active = history[:, -1, :, 9] > 0
            self.memory_keys = features[active].detach()
            self.memory_residuals = (arrays["targets"][index, 0, :, 2:4] - state[..., 2:4])[
                active
            ].detach()
        self.memory_active = {"split": "train", "trajectory_ids": arrays["ids"]}
