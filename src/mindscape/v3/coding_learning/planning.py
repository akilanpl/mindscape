"""Learned visible-test usefulness and runtime predictions; no private-test access."""

import difflib
import math

import numpy as np
import torch
from torch import nn

from .concepts import FEATURES, extract, similarity, vector
from .execution import safe_source

DIM = 3 * len(FEATURES) + 7


def features(repository, proposal, contract, memory=(), concepts=True):
    before = extract(repository, contract)
    try:
        after_repo = proposal.apply(repository)
    except ValueError:
        after_repo = repository
    after = extract(after_repo, contract)
    x, y = vector(before), vector(after)
    if not concepts:
        x = x * 0
        y = y * 0
    changed = sum(repository.get(k) != v for k, v in after_repo.items())
    a = "\n".join(repository.values())
    b = "\n".join(after_repo.values())
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    try:
        safe_source(after_repo)
        syntax = 1.0
    except (ValueError, SyntaxError):
        syntax = 0.0
    support = (
        max((similarity(after, r["after_concepts"]) for r in memory), default=0.0)
        if concepts
        else 0.0
    )
    return np.concatenate(
        (
            x,
            y,
            y - x,
            np.asarray(
                [
                    changed,
                    ratio,
                    len(b) / 16000,
                    syntax,
                    support,
                    float(bool(memory)),
                    float(a == b),
                ],
                dtype=np.float32,
            ),
        )
    )


class ProposalEvaluator(nn.Module):
    def __init__(self, seed=11):
        super().__init__()
        torch.manual_seed(seed)
        self.network = nn.Sequential(nn.Linear(DIM, 32), nn.Tanh(), nn.Linear(32, 2))

    def predict(self, x):
        with torch.inference_mode():
            z = self(torch.as_tensor(x, dtype=torch.float32))
            return {
                "visible_usefulness": float(torch.sigmoid(z[0])),
                "seconds": float(torch.expm1(z[1].clamp(0, 4))),
                "kind": "hypothetical_unverified",
                "private_success": None,
            }

    def forward(self, x):
        return self.network(x)


def fit(rows, seed, updates=400):
    if not rows or any(r["split"] != "train" for r in rows):
        raise ValueError("Training evidence only")
    torch.set_num_threads(2)
    model = ProposalEvaluator(seed)
    x = torch.tensor(np.asarray([r["features"] for r in rows]), dtype=torch.float32)
    labels = torch.tensor([[float(r["visible_pass"]), math.log1p(r["seconds"])] for r in rows])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.004)
    history = []
    for step in range(updates):
        optimizer.zero_grad()
        out = model(x)
        loss = nn.functional.binary_cross_entropy_with_logits(
            out[:, 0], labels[:, 0]
        ) + nn.functional.mse_loss(out[:, 1], labels[:, 1])
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == updates - 1:
            history.append({"update": step + 1, "loss": float(loss.detach())})
    return model, {
        "seed": seed,
        "updates": updates,
        "rows": len(rows),
        "history": history,
        "positive_rows": sum(r["visible_pass"] for r in rows),
        "parameters": sum(p.numel() for p in model.parameters()),
    }


def choose(repository, proposals, contract, memory, model, planning=True, concepts=True):
    predictions = [
        model.predict(features(repository, p, contract, memory, concepts)) for p in proposals
    ]
    # The scorer has never received a held-out execution or a private-test label.
    index = (
        max(
            range(len(proposals)),
            key=lambda i: predictions[i]["visible_usefulness"] - 0.01 * predictions[i]["seconds"],
        )
        if planning
        else 0
    )
    return index, predictions
