"""No arithmetic teacher, target, verifier or demonstration memory in neural inference."""

import torch
from torch import nn

from mindscape.v3.physics_world.accelerator import require_mps, verify_placement

from .arithmetic import OPS, sign_features


def features(state):
    a, b, c = state["left"], state["right"], state["incoming"]
    # Decimal embedding indices plus non-answer relational observations. No sum/product feature.
    relation = [a / 9, b / 9, c / 9, float(a == b), float(a < b), float(a == 0), float(b == 0)]
    return [OPS.index(state["operation"]), a, b, c], relation


class NumericTransition(nn.Module):
    def __init__(self, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.operation = nn.Embedding(3, 8)
        self.digit = nn.Embedding(10, 12)
        self.relation = nn.Sequential(nn.Linear(7, 16), nn.Tanh())
        self.transition = nn.GRUCell(60, 48)
        self.digit_head = nn.Linear(48, 10)
        self.carry_head = nn.Linear(48, 10)
        self.sign_head = nn.Sequential(nn.Linear(9, 32), nn.Tanh(), nn.Linear(32, 2))

    def forward(self, indices, relations, hidden):
        verify_placement(self, indices, relations, hidden)
        x = torch.cat(
            (
                self.operation(indices[:, 0]),
                *[self.digit(indices[:, i]) for i in (1, 2, 3)],
                self.relation(relations),
            ),
            1,
        )
        h = self.transition(x, hidden)
        return self.digit_head(h), self.carry_head(h), h


class Neural:
    def __init__(self, model, kind="recurrent"):
        require_mps()
        self.model = model.eval()
        self.kind = kind
        self.calls = 0
        self.reset()

    def reset(self):
        self.hidden = torch.zeros((1, 48), device="mps")

    def local(self, state):
        indices, relation = features(state)
        h = self.hidden if self.kind == "recurrent" else torch.zeros_like(self.hidden)
        with torch.inference_mode():
            d, c, self.hidden = self.model(
                torch.tensor([indices], device="mps"), torch.tensor([relation], device="mps"), h
            )
            dp, cp = d.softmax(-1)[0], c.softmax(-1)[0]
        self.calls += 1
        return (
            int(d.argmax(-1).item()),
            int(c.argmax(-1).item()),
            {
                "kind": "neural_unverified",
                "digit_probabilities": dp.cpu().tolist(),
                "carry_probabilities": cp.cpu().tolist(),
            },
        )

    def sign(self, a, b, op):
        with torch.inference_mode():
            z = self.model.sign_head(torch.tensor([sign_features(a, b, op)], device="mps"))
        self.calls += 1
        return int(z.argmax(-1).item()), {
            "kind": "neural_unverified",
            "probabilities": z.softmax(-1)[0].cpu().tolist(),
        }
