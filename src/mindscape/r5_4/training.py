"""Bounded matched schedule. Only this module creates an optimizer."""
import random
import time
import torch
from torch.nn import functional as F
from .data import operand, reference
from .inference import columns


def objective(model, tasks, update):
    device = next(model.parameters()).device
    traces = [reference(operand(t)) for t in tasks]
    losses, exposure = [], set()
    # First half supervised; second half combines TF with own-predicted-carry trajectories.
    for free in (False, True) if update > 400 else (False,):
        carry = torch.zeros(len(tasks), device=device, dtype=torch.long)
        hidden, numerator, denominator = None, torch.zeros((), device=device), 0
        for position in range(max(map(len, traces))):
            active = [i for i, tr in enumerate(traces) if position < len(tr)]
            left = torch.tensor([tr[position][0] if position < len(tr) else 0 for tr in traces], device=device)
            right = torch.tensor([tr[position][1] if position < len(tr) else 0 for tr in traces], device=device)
            if not free:
                carry = torch.tensor([tr[position][2] if position < len(tr) else 0 for tr in traces], device=device)
            for i in active:
                exposure.add((int(left[i].item()), int(right[i].item()), int(carry[i].item())))
            # Training-only local supervision for the actually visited input, including wrong carries.
            total = left + right + carry
            digit_logits, carry_logits, hidden = model(left, right, carry, hidden)
            idx = torch.tensor(active, device=device)
            numerator = numerator + F.cross_entropy(digit_logits[idx], (total % 10)[idx], reduction='sum')
            numerator = numerator + F.cross_entropy(carry_logits[idx], (total // 10)[idx], reduction='sum')
            denominator += len(active)
            carry = carry_logits.detach().argmax(-1)  # no gradient through hard choice; no correction
        losses.append(numerator / denominator)
    return torch.stack(losses).mean(), exposure


def fit(model, tasks, seed, log_path):
    rng = random.Random(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0001)
    start, exposures, progress = time.monotonic(), set(), []
    order = []
    for update in range(1, 801):
        if time.monotonic() - start > 900:
            raise RuntimeError('Fit exceeded frozen 900-second budget; no retry')
        if (update - 1) % 32 == 0:
            order = list(range(len(tasks)))
            rng.shuffle(order)
        offset = ((update - 1) % 32) * 32
        batch = [tasks[i] for i in order[offset:offset + 32]]
        optimizer.zero_grad(set_to_none=True)
        loss, visited = objective(model, batch, update)
        if not torch.isfinite(loss).item():
            raise RuntimeError('Nonfinite loss')
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all().item() for p in model.parameters()):
            raise RuntimeError('Nonfinite/missing gradient')
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.mps.synchronize()
        exposures |= visited
        progress.append(dict(update=update, loss=float(loss.item()), dtype=str(loss.dtype)))
        # Persistent progress makes partial failures independently visible.
        import json
        with log_path.open('a') as stream:
            stream.write(json.dumps(progress[-1], sort_keys=True) + '\n')
    if any(not torch.isfinite(p).all().item() for p in model.parameters()):
        raise RuntimeError('Nonfinite checkpoint')
    return optimizer, progress, sorted(exposures)
