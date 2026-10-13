"""Only learned digit/carry logits; inputs encode no sum, position or answer."""
import torch
from torch import nn
from torch.nn import functional as F

COUNTS = {'markov': 2188, 'recurrent': 2124}


def features(left, right, carry):
    return torch.cat([F.one_hot(left, 10), F.one_hot(right, 10),
                      F.one_hot(carry, 2)], dim=-1).to(torch.float32)


class Transition(nn.Module):
    def __init__(self, kind):
        super().__init__()
        if kind not in COUNTS:
            raise ValueError(kind)
        self.kind = kind
        if kind == 'markov':
            self.cell = nn.Sequential(nn.Linear(22, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh())
            self.head = nn.Linear(32, 12)
        else:
            self.cell = nn.GRUCell(22, 16)
            self.head = nn.Linear(16, 12)
        assert sum(p.numel() for p in self.parameters()) == COUNTS[kind]

    def forward(self, left, right, carry, hidden=None):
        x = features(left, right, carry)
        if self.kind == 'markov':
            hidden = self.cell(x)
        else:
            if hidden is None:
                hidden = torch.zeros((x.shape[0], 16), device=x.device, dtype=x.dtype)
            hidden = self.cell(x, hidden)
        logits = self.head(hidden)
        return logits[:, :10], logits[:, 10:], hidden if self.kind == 'recurrent' else None


def adapter(model):
    device = next(model.parameters()).device

    @torch.no_grad()
    def predict(left, right, carry, hidden):
        tensors = [torch.tensor([v], device=device, dtype=torch.long) for v in (left, right, carry)]
        digit_logits, carry_logits, state = model(*tensors, hidden)
        return int(digit_logits.argmax(-1).item()), int(carry_logits.argmax(-1).item()), state
    return predict
