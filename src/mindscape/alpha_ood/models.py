"""Matched-feature local baselines. No training loop, dataset access or launch side effect."""
import torch
from torch import nn

from mindscape.r5_2_math.model import features


class LocalTransition(nn.Module):
    """Declared arithmetic scaffold retained; arithmetic rule is never an input feature.

    Recurrent core has 15,840 parameters, feedforward core 15,744 (0.61% fewer).
    All common embeddings/heads and final-only update/exposure budgets are matched.
    Core capacity is approximately matched, not falsely claimed identical.
    """
    def __init__(self, kind: str, seed: int):
        super().__init__()
        if kind not in ("feedforward", "recurrent"):
            raise ValueError("unsupported learned condition")
        self.kind = kind
        # Fork the RNG so fixture creation does not change another campaign RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.operation = nn.Embedding(3, 8)
            self.digit = nn.Embedding(10, 12)
            self.relation = nn.Sequential(nn.Linear(7, 16), nn.Tanh())
            self.digit_head = nn.Linear(48, 10)
            self.carry_head = nn.Linear(48, 10)
            self.sign_head = nn.Sequential(nn.Linear(9, 32), nn.Tanh(), nn.Linear(32, 2))
            self.core = (nn.GRUCell(60, 48) if kind == "recurrent" else
                         nn.Sequential(nn.Linear(60, 144), nn.Tanh(), nn.Linear(144, 48), nn.Tanh()))

    def forward(self, indices, relations, hidden):
        x = torch.cat((self.operation(indices[:, 0]),
                       *[self.digit(indices[:, i]) for i in (1, 2, 3)],
                       self.relation(relations)), dim=1)
        h = self.core(x, hidden) if self.kind == "recurrent" else self.core(x)
        return self.digit_head(h), self.carry_head(h), h

    def parameter_receipt(self):
        return {"condition": self.kind,
                "parameters": sum(p.numel() for p in self.parameters()),
                "core_parameters": sum(p.numel() for p in self.core.parameters()),
                "features": "R5.2 observable digit/operation/incoming and relational features"}


class ScaffoldPredictor:
    """Adapter for existing R5.2 write-only arithmetic.run; author never uses OOD here."""
    def __init__(self, model, device):
        self.model = model.eval()
        self.device = device
        self.reset()

    def reset(self):
        self.hidden = torch.zeros((1, 48), device=self.device)

    def local(self, state):
        indices, relations = features(state)
        with torch.inference_mode():
            d, c, self.hidden = self.model(torch.tensor([indices], device=self.device),
                                          torch.tensor([relations], device=self.device), self.hidden)
        return int(d.argmax(-1).item()), int(c.argmax(-1).item()), {"kind": self.model.kind}

    def sign(self, a, b, op):
        from mindscape.r5_2_math.arithmetic import sign_features
        with torch.inference_mode():
            logits = self.model.sign_head(torch.tensor([sign_features(a, b, op)], device=self.device))
        return int(logits.argmax(-1).item()), {"kind": self.model.kind}
