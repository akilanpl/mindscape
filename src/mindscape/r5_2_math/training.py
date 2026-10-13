"""MPS fits from frozen training demonstrations only; no OOD/development optimization."""

import hashlib
import json
import time

import numpy as np
import torch
from torch import nn

from mindscape.v3.physics_world.accelerator import require_mps, verify_placement

from .arithmetic import sign_features
from .model import NumericTransition, features


def train_examples(tasks, traces, held):
    sequences = []
    signs = []
    memory = {}
    sign_memory = {}
    held_exposure = 0
    for task, trace in zip(tasks, traces, strict=True):
        sequence = []
        # Preserve the exact teacher path; no target feedback is used at evaluation.
        for e in trace["transitions"]:
            state = e["state"]
            key = (state["operation"], state["left"], state["right"], state["incoming"])
            label = [e["written_digit"], e["written_carry"]]
            indices, relation = features(state)
            allowed = key not in held
            held_exposure += not allowed
            sequence.append((indices, relation, label, float(allowed), state["index"] == 0))
            if allowed:
                memory["|".join(map(str, key))] = label
        sequences.append(sequence)
        current = task["operands"][0]
        for op, b, stage in zip(
            task["operations"], task["operands"][1:], trace["stages"], strict=True
        ):
            signs.append((sign_features(current, b, op), stage["sign"]))
            sign_memory[f"{current}|{b}|{op}"] = stage["sign"]
            current = int(stage["answer"])
    return sequences, signs, memory, sign_memory, held_exposure


def fit(sequences, signs, kind, seed, protocol, root):
    require_mps()
    with torch.device("mps"):
        model = NumericTransition(seed)
    verify_placement(model)
    opt = torch.optim.Adam(model.parameters(), lr=protocol["learning_rate"])
    rng = np.random.default_rng(seed)
    losses = []
    start = time.perf_counter()
    max_length = max(len(s) for s in sequences)
    xs = torch.zeros((len(sequences), max_length, 4), device="mps", dtype=torch.long)
    rs = torch.zeros((len(sequences), max_length, 7), device="mps")
    ys = torch.zeros((len(sequences), max_length, 2), device="mps", dtype=torch.long)
    mask = torch.zeros((len(sequences), max_length), device="mps")
    reset = torch.zeros_like(mask)
    for i, sequence in enumerate(sequences):
        for j, (x, r, y, m, boundary) in enumerate(sequence):
            xs[i, j] = torch.tensor(x, device="mps")
            rs[i, j] = torch.tensor(r, device="mps")
            ys[i, j] = torch.tensor(y, device="mps")
            mask[i, j] = m
            # Each scheduled row/addition starts at index0; training resets exactly like inference.
            # Index carried separately in prebuilt trace order below.
    # Identify row boundaries through explicit feature packaging rather than numeric targets.
    for i, sequence in enumerate(sequences):
        for j, entry in enumerate(sequence):
            reset[i, j] = float(entry[4]) if len(entry) > 4 else float(j == 0)
    signx = torch.tensor([s[0] for s in signs], device="mps")
    signy = torch.tensor([s[1] for s in signs], device="mps")
    for update in range(protocol["updates_per_fit"]):
        idx = torch.tensor(rng.integers(len(sequences), size=protocol["batch_size"]), device="mps")
        si = torch.tensor(rng.integers(len(signs), size=protocol["batch_size"]), device="mps")
        hidden = torch.zeros((protocol["batch_size"], 48), device="mps")
        total = torch.zeros((), device="mps")
        denom = mask[idx].sum().clamp(min=1)
        opt.zero_grad(set_to_none=True)
        for j in range(max_length):
            hidden = (
                hidden * (1 - reset[idx, j, None])
                if kind == "recurrent"
                else torch.zeros_like(hidden)
            )
            d, c, hidden = model(xs[idx, j], rs[idx, j], hidden)
            loss = nn.functional.cross_entropy(
                d, ys[idx, j, 0], reduction="none"
            ) + nn.functional.cross_entropy(c, ys[idx, j, 1], reduction="none")
            total += (loss * mask[idx, j]).sum() / denom
        signloss = nn.functional.cross_entropy(model.sign_head(signx[si]), signy[si])
        total += signloss
        if not torch.isfinite(total):
            raise ValueError("Nonfinite loss")
        total.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("Nonfinite/disconnected gradient")
        opt.step()
        losses.append(float(total.detach().item()))
        with (root / f"{kind}-s{seed}-progress.jsonl").open("a") as stream:
            stream.write(json.dumps({"update": update + 1, "loss": losses[-1]}) + "\n")
        if (update + 1) % 200 == 0:
            print(
                json.dumps({"kind": kind, "seed": seed, "update": update + 1, "loss": losses[-1]}),
                flush=True,
            )
    torch.mps.synchronize()
    path = root / f"{kind}-s{seed}.pt"
    torch.save(
        {
            "state": model.state_dict(),
            "optimizer": opt.state_dict(),
            "seed": seed,
            "kind": kind,
            "updates": protocol["updates_per_fit"],
            "losses": losses,
        },
        path,
    )
    return model, {
        "kind": kind,
        "seed": seed,
        "parameters": sum(p.numel() for p in model.parameters()),
        "updates": protocol["updates_per_fit"],
        "seconds": time.perf_counter() - start,
        "losses": losses,
        "checkpoint": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
