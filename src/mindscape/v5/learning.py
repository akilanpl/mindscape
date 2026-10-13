"""Matched small MPS fits; no simulator, AST executor or qualification labels in optimizer."""

import hashlib
import json
import time

import numpy as np
import torch
from torch import nn

from mindscape.v3.physics_world.accelerator import require_mps

from .model import GraphTransition, tensors
from .schema import MAX_NODES, OUTPUTS


def targets(rows):
    y = torch.zeros((len(rows), MAX_NODES, OUTPUTS[rows[0]["domain"]]), device="mps")
    for i, r in enumerate(rows):
        a = r["target"]["delta"]
        y[i, : len(a)] = torch.tensor(a, device="mps")
    valid = torch.tensor([r["target"]["valid"] for r in rows], device="mps", dtype=torch.float32)
    return y, valid


def prediction(model, rows):
    started = time.perf_counter()
    records = []
    for domain in ("mathematics", "coding", "physics"):
        selected = [r for r in rows if r["domain"] == domain]
        if not selected:
            continue
        with torch.inference_mode():
            delta, valid = model(*tensors([r["graph"] for r in selected]))
            deltas = delta.cpu().tolist()
            prob = valid.sigmoid().cpu().tolist()
        for r, d, p in zip(selected, deltas, prob):
            records.append(
                {
                    "identity": r["identity"],
                    "delta": d[: len(r["graph"]["entities"])],
                    "valid_probability": p,
                }
            )
    torch.mps.synchronize()
    return records, time.perf_counter() - started


def metric(rows, predictions):
    by = {p["identity"]: p for p in predictions}
    errors = []
    invalid = []
    class_correct = {0: [], 1: []}
    integer = []
    task_correct = []
    kinematic = []
    for row in rows:
        p = by[row["identity"]]
        target = row["target"]
        valid = target["valid"]
        validity_correct = (p["valid_probability"] >= 0.5) == bool(valid)
        class_correct[valid].append(int(validity_correct))
        if row["domain"] in ("mathematics", "coding"):
            correct = all(
                round(x * 100) == round(y * 100)
                for a, b in zip(p["delta"], target["delta"], strict=True)
                for x, y in zip(a, b, strict=True)
            )
        else:
            correct = all(
                abs(x - y) * 4 <= 0.25
                for a, b in zip(p["delta"], target["delta"], strict=True)
                for x, y in zip(a, b, strict=True)
            )
            if valid:
                for before, delta in zip(target["before"], p["delta"], strict=True):
                    kinematic.extend(
                        abs(delta[i] * 4 - (before[i + 2] + delta[i + 2] * 4) / 30) for i in (0, 1)
                    )
        task_correct.append(int(correct and validity_correct))
        for a, b in zip(p["delta"], target["delta"], strict=True):
            for x, y in zip(a, b, strict=True):
                if not np.isfinite(x):
                    raise ValueError("Nonfinite prediction")
                errors.append((x - y) ** 2)
                if not valid:
                    invalid.append(x * x)
                if row["domain"] in ("mathematics", "coding"):
                    integer.append(abs(x * 100 - round(x * 100)) <= 0.1)
    return {
        "tasks": len(rows),
        "normalized_mse": float(np.mean(errors)),
        "validity_balanced_accuracy": float(
            np.mean([np.mean(v) for v in class_correct.values() if v])
        ),
        "invalid_noop_mse": float(np.mean(invalid)) if invalid else None,
        "integer_consistency_fraction": float(np.mean(integer)) if integer else None,
        "transition_task_correct": sum(task_correct),
        "transition_task_accuracy": float(np.mean(task_correct)),
        "mean_physics_position_velocity_constraint_error": float(np.mean(kinematic))
        if kinematic
        else None,
        "finite": True,
    }


def fit(rows, target, condition, seed, protocol, path):
    require_mps()
    with torch.device("mps"):
        model = GraphTransition(seed)
    opt = torch.optim.Adam(model.parameters(), lr=protocol["learning_rate"])
    target_rows = [r for r in rows if r["domain"] == target and r["split"] == "train"][
        : protocol["target_training_tasks"]
    ]
    sources = [d for d in ("mathematics", "coding", "physics") if d != target]
    batches = {}
    for d in ("mathematics", "coding", "physics"):
        subset = (
            target_rows
            if d == target
            else [r for r in rows if r["domain"] == d and r["split"] == "train"]
        )
        inputs = tensors([r["graph"] for r in subset])
        delta, valid = targets(subset)
        if condition == "randomized_auxiliary" and d != target:
            rng = np.random.default_rng(seed + sum(map(ord, d)))
            order = torch.tensor(rng.permutation(len(subset)), device="mps")
            delta, valid = delta[order], valid[order]
        batches[d] = inputs, delta, valid
    initial, _ = prediction(model, target_rows)
    initial_metric = metric(target_rows, initial)
    losses = []
    curves = []
    start = time.perf_counter()
    pre = protocol["updates"]["auxiliary_or_target"]
    total = pre + protocol["updates"]["target"]
    for step in range(total):
        d = sources[step % 2] if step < pre and condition != "domain_only" else target
        inputs, delta, valid = batches[d]
        rng = np.random.default_rng(seed * 10000 + step)
        indices = torch.tensor(rng.integers(len(valid), size=protocol["batch_size"]), device="mps")
        domain, *xx = inputs
        selected = [x[indices] for x in xx]
        opt.zero_grad(set_to_none=True)
        effect, precondition = model(domain, *selected)
        mask = selected[3][..., None].expand_as(effect)
        loss = (
            ((effect - delta[indices]) ** 2) * mask
        ).sum() / mask.sum() + 0.1 * nn.functional.binary_cross_entropy_with_logits(
            precondition, valid[indices]
        )
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss")
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().item()))
        if step + 1 in (pre, pre + 30, total):
            predictions, _ = prediction(model, target_rows)
            curves.append({"update": step + 1, "target_training": metric(target_rows, predictions)})
    torch.mps.synchronize()
    seconds = time.perf_counter() - start
    identity = {"target": target, "condition": condition, "seed": seed}
    torch.save(
        {
            "identity": identity,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "step": total,
            "losses": losses,
        },
        path,
    )
    record = {
        "identity": identity,
        "parameters": sum(p.numel() for p in model.parameters()),
        "updates": total,
        "graph_presentations": total * protocol["batch_size"],
        "seconds": seconds,
        "losses": losses,
        "curves": curves,
        "initial_target_training": initial_metric,
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    return model, record


def run(rows, sealed, protocol, root):
    root.mkdir(exist_ok=False)
    train = [r for r in rows if r["split"] == "train"]
    evalrows = sealed
    results = []
    for target in ("mathematics", "coding", "physics"):
        for seed in protocol["optimizer_seeds"]:
            for condition in ("domain_only", "shared", "randomized_auxiliary"):
                path = root / f"{target}-{condition}-s{seed}.pt"
                model, record = fit(train, target, condition, seed, protocol, path)
                predicted, elapsed = prediction(model, evalrows)
                subset = [r for r in evalrows if r["domain"] == target]
                record["predictions"] = [
                    p for p in predicted if p["identity"] in {r["identity"] for r in subset}
                ]
                record["composition"] = metric(subset, record["predictions"])
                record["inference_seconds_all_domains"] = elapsed
                results.append(record)
                (path.with_suffix(".json")).write_text(json.dumps(record, indent=2) + "\n")
                print(
                    json.dumps({"fit": record["identity"], "composition": record["composition"]}),
                    flush=True,
                )
    baselines = {}
    for d in ("mathematics", "coding", "physics"):
        selected = [r for r in evalrows if r["domain"] == d]
        predictions = [
            {
                "identity": r["identity"],
                "delta": [[0.0] * len(x) for x in r["target"]["delta"]],
                "valid_probability": 0.5,
            }
            for r in selected
        ]
        baselines[d] = {
            "persistence": metric(selected, predictions),
            "procedural": {
                "normalized_mse": 0.0,
                "validity_balanced_accuracy": 1.0,
                "tasks": len(selected),
                "meaning": "Independent executed target defines expert reference ceiling; no learned result.",
            },
        }
    by = {
        (r["identity"]["target"], r["identity"]["condition"], r["identity"]["seed"]): r
        for r in results
    }
    gains = {}
    for d in ("mathematics", "coding", "physics"):
        gains[d] = {}
        for control in ("domain_only", "randomized_auxiliary"):
            gains[d][control] = [
                (
                    by[d, control, s]["composition"]["normalized_mse"]
                    - by[d, "shared", s]["composition"]["normalized_mse"]
                )
                / max(1e-12, by[d, control, s]["composition"]["normalized_mse"])
                for s in protocol["optimizer_seeds"]
            ]
    gate = protocol["scientific_gates"]
    shared = [
        r
        for r in results
        if r["identity"]["target"] == "coding" and r["identity"]["condition"] == "shared"
    ]
    checks = {
        "primary_control_gains": all(
            np.mean(v) >= gate["primary_relative_mse_improvement_vs_each_control"]
            for v in gains["coding"].values()
        ),
        "seed_consistency": all(v > 0 for values in gains["coding"].values() for v in values),
        "persistence_gain": np.mean([r["composition"]["normalized_mse"] for r in shared])
        <= (1 - gate["primary_relative_improvement_over_persistence"])
        * baselines["coding"]["persistence"]["normalized_mse"],
        "validity": all(
            r["composition"]["validity_balanced_accuracy"]
            >= gate["primary_validity_balanced_accuracy"]
            for r in shared
        ),
        "no_large_negative_transfer": all(
            np.mean(gains[d]["domain_only"])
            >= -gate["maximum_mean_other_domain_relative_mse_regression"]
            for d in ("mathematics", "physics")
        ),
        "finite": all(r["composition"]["finite"] for r in results),
    }
    return {
        "decision": "GO_TO_INDEPENDENT_CONFIRMATION" if all(checks.values()) else "NO_GO",
        "gates": {k: bool(v) for k, v in checks.items()},
        "fits": results,
        "baselines": baselines,
        "relative_gains": gains,
        "independent_tasks_per_domain": 12,
        "seeds": 3,
        "limitations": [
            "Bounded synthetic microdomains, not specialist repair/planning or general reasoning.",
            "Composition tasks shared across seeds; seeds are repeated fits, not extra independent tasks.",
            "Procedural expert is perfect; gains claimed only against learned controls and persistence.",
            "Fully observed one-step physics and no contact/spring interactions; held-out variation is object count.",
            "Inputs/heads differ by domain; latent sharing does not assert semantic identity of entities.",
        ],
    }
