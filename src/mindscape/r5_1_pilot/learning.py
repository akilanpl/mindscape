"""MPS-only 120-update fits. Graph inference has no target or changed mask argument."""

import hashlib
import json
import time

import numpy as np
import torch
from torch import nn

from mindscape.v3.physics_world.accelerator import require_mps, verify_placement
from mindscape.v3.schema import digest
from mindscape.v5.model import GraphTransition, tensors
from mindscape.v5.schema import MAX_NODES, OUTPUTS

from .scoring import metrics


def state_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        a = value.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(tuple(a.shape)).encode())
        h.update(str(a.dtype).encode())
        h.update(a.numpy().tobytes())
    return h.hexdigest()


def predict(model, graphs):
    if not graphs:
        raise ValueError("No public graphs")
    graphs[0]["domain"]
    started = time.perf_counter()
    with torch.inference_mode():
        effect, valid = model(*tensors(graphs))
        effect = effect.cpu().tolist()
        valid = valid.sigmoid().cpu().tolist()
    return [
        {"delta": d[: len(g["entities"])], "valid_probability": v}
        for g, d, v in zip(graphs, effect, valid, strict=True)
    ], time.perf_counter() - started


def predictions(model, rows):
    p, elapsed = predict(model, [r["graph"] for r in rows])
    return [{"identity": r["identity"], **v} for r, v in zip(rows, p, strict=True)], elapsed


def label_tensors(rows):
    d = rows[0]["domain"]
    y = torch.zeros((len(rows), MAX_NODES, OUTPUTS[d]), device="mps")
    for i, row in enumerate(rows):
        values = row["target"]["delta"]
        y[i, : len(values)] = torch.tensor(values, device="mps")
    return y, torch.tensor([r["target"]["valid"] for r in rows], device="mps", dtype=torch.float32)


def effect_loss(effect, target, active):
    errors = (effect - target).square().mean(-1)
    changed = (target.abs().amax(-1) > 1e-8) * active
    unchanged = (1 - changed) * active
    ce = (errors * changed).sum(1) / changed.sum(1).clamp(min=1)
    ue = (errors * unchanged).sum(1) / unchanged.sum(1).clamp(min=1)
    return (0.5 * ce + 0.5 * ue).mean()


def fit(rows, target, condition, seed, protocol, freeze_hash, root):
    require_mps()
    if any(r["split"] != "development" for r in rows):
        raise ValueError("Development labels only")
    sources = [d for d in ("mathematics", "coding", "physics") if d != target]
    with torch.device("mps"):
        model = GraphTransition(seed)
    verify_placement(model)
    if sum(p.numel() for p in model.parameters()) != 6967:
        raise ValueError("Architecture changed")
    initial_hash = state_hash(model.state_dict())
    opt = torch.optim.Adam(
        model.parameters(),
        lr=protocol["optimizer"]["learning_rate"],
        betas=tuple(protocol["optimizer"]["betas"]),
        eps=protocol["optimizer"]["epsilon"],
        weight_decay=0,
    )
    blocks = {}
    permutations = {}
    for domain in ("mathematics", "coding", "physics"):
        selected = [r for r in rows if r["domain"] == domain]
        if len(selected) != 18:
            raise ValueError("Development coverage changed")
        x = tensors([r["graph"] for r in selected])
        y, v = label_tensors(selected)
        if condition == "randomized_auxiliary" and domain != target:
            order = (
                np.random.default_rng(seed + sum(map(ord, domain)))
                .permutation(len(selected))
                .tolist()
            )
            index = torch.tensor(order, device="mps")
            y = y[index]
            v = v[index]
            permutations[domain] = {
                "order": order,
                "original_ids": [r["identity"] for r in selected],
                "target_ids": [selected[i]["identity"] for i in order],
                "original_target_hash": digest([r["target"] for r in selected]),
                "permuted_target_hash": digest([selected[i]["target"] for i in order]),
                "fixed_identity_associations": sum(i == j for i, j in enumerate(order)),
                "identical_effect_and_validity_associations": sum(
                    selected[i]["target"]["delta"] == selected[j]["target"]["delta"]
                    and selected[i]["target"]["valid"] == selected[j]["target"]["valid"]
                    for i, j in enumerate(order)
                ),
            }
        blocks[domain] = (x, y, v)
    target_rows = [r for r in rows if r["domain"] == target]
    initial_predictions, _ = predictions(model, target_rows)
    initial_metric, _ = metrics(target_rows, initial_predictions)
    curves = [{"update": 0, "metrics": initial_metric, "predictions": initial_predictions}]
    losses = []
    checkpoints = []
    counts = {d: 0 for d in ("mathematics", "coding", "physics")}
    steps = []
    elapsed_start = time.perf_counter()
    checkpoint_times = []
    identity = {
        "target": target,
        "condition": condition,
        "seed": seed,
        "protocol_hash": digest(protocol),
        "development_hash": digest(rows),
        "source_freeze_hash": freeze_hash,
    }
    first_input_labels = {r["identity"]: digest(r["target"]) for r in rows}
    for step in range(120):
        (root / "active-fit.json").write_text(
            json.dumps({"identity": identity, "next_update": step + 1}) + "\n"
        )
        domain = sources[step % 2] if step < 60 and condition != "domain_only" else target
        x, y, v = blocks[domain]
        indices = torch.tensor(
            np.random.default_rng(seed * 10000 + step).integers(len(v), size=16), device="mps"
        )
        name, *xx = x
        selected = [a[indices] for a in xx]
        opt.zero_grad(set_to_none=True)
        effect, valid = model(name, *selected)
        el = effect_loss(effect, y[indices], selected[3])
        vl = nn.functional.binary_cross_entropy_with_logits(valid, v[indices])
        loss = el + 0.1 * vl
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss")
        loss.backward()
        for label, p in model.named_parameters():
            expected = label.startswith(
                ("core.", "encoders." + domain + ".", "heads." + domain + ".")
            )
            if expected and (
                p.grad is None or p.grad.device.type != "mps" or not torch.isfinite(p.grad).all()
            ):
                raise ValueError("Missing/nonfinite active gradient: " + label)
            if not expected and p.grad is not None:
                raise ValueError("Inactive domain received gradient: " + label)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        opt.step()
        verify_placement(model)
        change = {
            group: float(
                sum(
                    (p - before[n]).square().sum().item()
                    for n, p in model.named_parameters()
                    if n.startswith(prefix)
                )
            )
            ** 0.5
            for group, prefix in [
                ("core", "core."),
                ("encoder", "encoders." + domain + "."),
                ("head", "heads." + domain + "."),
            ]
        }
        for n, p in model.named_parameters():
            if not n.startswith(
                ("core.", "encoders." + domain + ".", "heads." + domain + ".")
            ) and not torch.equal(before[n], p):
                raise ValueError("Inactive weights changed")
            if not torch.isfinite(p).all():
                raise ValueError("Nonfinite parameter")
        if any(value <= 0 for value in change.values()):
            raise ValueError("Active block did not learn/update")
        counts[domain] += 1
        steps.append({"update": step + 1, "domain": domain, "block_parameter_change_l2": change})
        losses.append(
            {
                "update": step + 1,
                "domain": domain,
                "effect_loss": float(el.detach().item()),
                "validity_loss": float(vl.detach().item()),
                "total_loss": float(loss.detach().item()),
            }
        )
        with (root / f"{target}-{condition}-s{seed}-progress.jsonl").open("a") as stream:
            stream.write(json.dumps({"loss": losses[-1], "parameter_changes": steps[-1]}) + "\n")
        if step + 1 in (60, 90, 120):
            pp, _ = predictions(model, target_rows)
            mm, _ = metrics(target_rows, pp)
            curves.append({"update": step + 1, "metrics": mm, "predictions": pp})
        if step + 1 in (60, 120):
            torch.mps.synchronize()
            start = time.perf_counter()
            path = root / f"{target}-{condition}-s{seed}-u{step + 1}.pt"
            torch.save(
                {
                    "identity": identity,
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "step": step + 1,
                    "losses": losses,
                    "domain_updates": counts,
                    "initial_state_hash": initial_hash,
                },
                path,
            )
            checkpoints.append(
                {
                    "file": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "update": step + 1,
                }
            )
            checkpoint_times.append(time.perf_counter() - start)
    if first_input_labels != {r["identity"]: digest(r["target"]) for r in rows}:
        raise ValueError("Training data mutated")
    record = {
        "identity": identity,
        "parameters": 6967,
        "device": "mps",
        "updates": 120,
        "graph_presentations": 1920,
        "initial_state_hash": initial_hash,
        "final_state_hash": state_hash(model.state_dict()),
        "domain_updates": counts,
        "gradient_audit_pass": True,
        "step_parameter_changes": steps,
        "auxiliary_permutations": permutations,
        "losses": losses,
        "curves": curves,
        "initial_training": curves[0]["metrics"],
        "final_training": curves[-1]["metrics"],
        "checkpoints": checkpoints,
        "seconds": time.perf_counter() - elapsed_start,
        "checkpoint_write_seconds": sum(checkpoint_times),
    }
    return model, record
