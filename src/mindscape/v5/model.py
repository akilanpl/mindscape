"""Shared action-conditioned relation processing; domain semantics live in encoders/heads."""

import torch
from torch import nn

from mindscape.v3.physics_world.accelerator import require_mps, verify_placement

from .schema import DIMS, MAX_NODES, OUTPUTS, validate


class TransitionCore(nn.Module):
    def __init__(self, width=24):
        super().__init__()
        self.action = nn.Linear(8, width)
        self.message = nn.Sequential(
            nn.Linear(3 * width + 3, width), nn.Tanh(), nn.Linear(width, width)
        )
        self.effect = nn.Sequential(nn.Linear(3 * width, width), nn.Tanh(), nn.Linear(width, width))
        self.precondition = nn.Sequential(nn.Linear(2 * width, 12), nn.Tanh(), nn.Linear(12, 1))

    def forward(self, h, relations, active, action):
        a = self.action(action)
        n = h.shape[1]
        for _ in range(2):
            receiver = h[:, :, None, :].expand(-1, -1, n, -1)
            sender = h[:, None, :, :].expand(-1, n, -1, -1)
            act = a[:, None, None, :].expand(-1, n, n, -1)
            message = self.message(torch.cat((receiver, sender, act, relations), -1))
            edge = (relations.abs().sum(-1) > 0) * active[:, :, None] * active[:, None, :]
            pooled = (message * edge[..., None]).sum(2) / edge.sum(2).clamp(min=1)[..., None]
            h = (
                h + self.effect(torch.cat((h, pooled, a[:, None, :].expand(-1, n, -1)), -1))
            ) * active[..., None]
        pooled = h.sum(1) / active.sum(1).clamp(min=1)[:, None]
        return h, self.precondition(torch.cat((pooled, a), -1)).squeeze(-1)


class GraphTransition(nn.Module):
    def __init__(self, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.encoders = nn.ModuleDict({d: nn.Linear(2 * n, 24) for d, n in DIMS.items()})
        self.core = TransitionCore()
        self.heads = nn.ModuleDict({d: nn.Linear(24, n) for d, n in OUTPUTS.items()})

    def forward(self, domain, x, masks, relations, active, action):
        verify_placement(self, x, masks, relations, active, action)
        h = torch.tanh(self.encoders[domain](torch.cat((x * masks, masks), -1))) * active[..., None]
        h, valid = self.core(h, relations, active, action)
        # No ground-truth precondition masking: all proposed actions produce uncorrected predictions.
        return self.heads[domain](h) * active[..., None], valid


def tensors(graphs):
    require_mps()
    for graph in graphs:
        validate(graph)
    domain = graphs[0]["domain"]
    if any(g["domain"] != domain for g in graphs):
        raise ValueError("Homogeneous domain batch required")
    batch = len(graphs)
    x = torch.zeros((batch, MAX_NODES, DIMS[domain]), device="mps")
    masks = torch.zeros_like(x)
    active = torch.zeros((batch, MAX_NODES), device="mps")
    relations = torch.zeros((batch, MAX_NODES, MAX_NODES, 3), device="mps")
    actions = []
    for i, g in enumerate(graphs):
        n = len(g["entities"])
        active[i, :n] = 1
        x[i, :n] = torch.tensor(g["entities"], device="mps")
        masks[i, :n] = torch.tensor(g["observed"], device="mps")
        relations[i, :n, :n] = torch.tensor(g["relations"], device="mps")
        actions.append(g["action"])
    return (
        domain,
        x,
        masks,
        relations,
        active,
        torch.tensor(actions, device="mps", dtype=torch.float32),
    )
