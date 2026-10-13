"""MPS-only matched small evaluators, with optimizer checkpoints and recorded losses."""

import hashlib
import json
import time

import numpy as np
import torch
from torch import nn

from mindscape.v3.physics_world.accelerator import require_mps, verify_placement
from mindscape.v3.schema import digest

from .data import DOMAINS
from .encoding import DIMS
from .scoring import metrics


class Evaluator(nn.Module):
    def __init__(self, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.encoders = nn.ModuleDict({d: nn.Linear(DIMS[d], 16) for d in DOMAINS})
        self.shared = nn.Sequential(nn.Tanh(), nn.Linear(16, 8), nn.Tanh(), nn.Linear(8, 1))

    def forward(self, domain, x):
        return self.shared(self.encoders[domain](x)).squeeze(-1)


def batches(rows, randomized, seed):
    result = []
    rng = np.random.default_rng(seed)
    for domain in DOMAINS:
        subset = [r for r in rows if r["domain"] == domain]
        if not subset:
            continue
        x = [v for r in subset for v in r["features"]]
        y = np.asarray([v for r in subset for v in r["labels"]], dtype=np.float32)
        if randomized:
            y = rng.permutation(y)
        result.append(
            (
                domain,
                torch.tensor(x, device="mps", dtype=torch.float32),
                torch.tensor(y, device="mps", dtype=torch.float32),
            )
        )
    return result


def predict(model, rows):
    records = []
    started = time.perf_counter()
    with torch.inference_mode():
        for row in rows:
            probs = (
                model(
                    row["domain"], torch.tensor(row["features"], device="mps", dtype=torch.float32)
                )
                .sigmoid()
                .cpu()
                .tolist()
            )
            records.append(
                {
                    "identity": row["identity"],
                    "probabilities": probs,
                    "selected": int(np.argmax(probs)),
                }
            )
    torch.mps.synchronize()
    return records, time.perf_counter() - started


def fit(rows, target, condition, seed, protocol, root):
    require_mps()
    path = root / f"{target}-{condition}-s{seed}.pt"
    target_rows = [r for r in rows if r["domain"] == target and r["split"] == "train"][
        : protocol["target_training_tasks"]
    ]
    aux = [r for r in rows if r["domain"] != target and r["split"] == "train"]
    pre = target_rows if condition == "domain_only" else aux
    first = batches(pre, condition == "randomized_auxiliary", seed)
    second = batches(target_rows, False, seed)
    identity = {
        "target": target,
        "condition": condition,
        "seed": seed,
        "rows": digest(rows),
        "protocol": digest(protocol),
    }
    with torch.device("mps"):
        model = Evaluator(seed)
    verify_placement(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=protocol["learning_rate"])
    initial, _ = predict(model, target_rows)
    initial_brier = metrics(target_rows, initial)["brier"]
    start, history, elapsed = 0, [], 0.0
    if path.exists():
        saved = torch.load(path, weights_only=True, map_location="mps")
        if saved["identity"] != identity:
            raise ValueError("Resume identity mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start, history, elapsed = saved["step"], saved["losses"], saved["seconds"]
    begin = time.perf_counter()
    pre_steps, target_steps = protocol["pretrain_updates"], protocol["target_updates"]
    for step in range(start, pre_steps + target_steps):
        batch = first if step < pre_steps else second
        d, x, y = batch[step % len(batch)]
        # Step-indexed RNG permits exact schedule reconstruction on resume.
        rng = np.random.default_rng(seed * 10000 + step)
        indices = torch.tensor(rng.integers(len(y), size=protocol["batch_size"]), device="mps")
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.binary_cross_entropy_with_logits(model(d, x[indices]), y[indices])
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss")
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().item()))
        if step + 1 in (pre_steps, pre_steps + target_steps):
            torch.mps.synchronize()
            temporary = path.with_suffix(".tmp")
            torch.save(
                {
                    "identity": identity,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step + 1,
                    "losses": history,
                    "seconds": elapsed + time.perf_counter() - begin,
                },
                temporary,
            )
            temporary.replace(path)
    torch.mps.synchronize()
    seconds = elapsed + time.perf_counter() - begin
    train_predictions, _ = predict(model, target_rows)
    final_brier = metrics(target_rows, train_predictions)["brier"]
    results = {}
    for split in ("iid", "ood"):
        subset = [r for r in rows if r["domain"] == target and r["split"] == split]
        predictions, inference_seconds = predict(model, subset)
        results[split] = {
            "metrics": metrics(subset, predictions),
            "predictions": predictions,
            "inference_seconds": inference_seconds,
        }
    return {
        "identity": identity,
        "results": results,
        "losses": history,
        "seconds": seconds,
        "parameters": sum(p.numel() for p in model.parameters()),
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "initial_target_training_brier": initial_brier,
        "final_target_training_brier": final_brier,
        "train_brier_improvement": initial_brier - final_brier,
        "train_brier_gain_over_constant": float(
            np.var([v for r in target_rows for v in r["labels"]])
        )
        - final_brier,
        "gradient_updates": pre_steps + target_steps,
        "candidate_presentations": (pre_steps + target_steps) * protocol["batch_size"],
        "target_label_tasks": len(target_rows),
        "auxiliary_label_tasks": 0 if condition == "domain_only" else len(aux),
    }


def run(rows, protocol, root):
    root.mkdir(exist_ok=True)
    records = []
    specialist = {}
    for target in DOMAINS:
        specialist[target] = {}
        for split in ("iid", "ood"):
            subset = [r for r in rows if r["domain"] == target and r["split"] == split]
            predictions = [
                {"identity": r["identity"], "selected": int(np.argmax(r["scores"]))} for r in subset
            ]
            specialist[target][split] = {
                "metrics": metrics(subset, predictions),
                "predictions": predictions,
            }
        for seed in protocol["seeds"]:
            for condition in ("domain_only", "shared", "randomized_auxiliary"):
                result = fit(rows, target, condition, seed, protocol, root)
                records.append(result)
                (root / f"{target}-{condition}-s{seed}.json").write_text(
                    json.dumps(result, indent=2)
                )
                print(
                    json.dumps(
                        {"fit": result["identity"], "ood": result["results"]["ood"]["metrics"]}
                    ),
                    flush=True,
                )
    gains = {}
    for target in DOMAINS:
        by = {
            (r["identity"]["seed"], r["identity"]["condition"]): r
            for r in records
            if r["identity"]["target"] == target
        }
        gains[target] = {
            control: [
                by[s, "shared"]["results"]["ood"]["metrics"]["accuracy"]
                - by[s, control]["results"]["ood"]["metrics"]["accuracy"]
                for s in protocol["seeds"]
            ]
            for control in ("domain_only", "randomized_auxiliary")
        }
    gate = protocol["gate"]
    primary = gains["coding"]
    checks = {
        "primary_gain": all(
            np.mean(v) * 100 >= gate["primary_gain_percentage_points"] for v in primary.values()
        ),
        "seed_consistency": all(v > 0 for values in primary.values() for v in values),
        "no_large_negative_transfer": all(
            np.mean(gains[d]["domain_only"]) * 100
            >= -gate["maximum_other_domain_regression_percentage_points"]
            for d in DOMAINS
        ),
        "candidate_availability": all(
            specialist[d]["ood"]["metrics"]["solvable"]
            >= gate["minimum_solvable_ood_tasks_per_domain"]
            for d in DOMAINS
        ),
        "learns_distinctions": all(
            r["train_brier_improvement"] >= gate["minimum_training_brier_improvement"]
            and r["train_brier_gain_over_constant"]
            >= gate["minimum_training_brier_improvement_over_constant"]
            for r in records
            if r["identity"]["condition"] == "shared"
        ),
        "mixed_target_training_labels": all(
            len(
                {
                    v
                    for r in [x for x in rows if x["domain"] == d and x["split"] == "train"][
                        : protocol["target_training_tasks"]
                    ]
                    for v in r["labels"]
                }
            )
            == 2
            for d in DOMAINS
        ),
    }
    summary = {
        "decision": "GO_TO_CONFIRMATION" if all(checks.values()) else "NO_GO",
        "gates": checks,
        "gains": gains,
        "specialist": specialist,
        "fits": records,
        "limitations": [
            "20 fresh development tasks per domain; only 8 OOD tasks",
            "Three optimizer seeds share candidate/task sets; not independent task replication",
            "Equal candidate presentations and updates; domain encoders differ in input size and wall time",
            "Math endpoint is local transition correctness, not whole-problem accuracy",
            "Source encoders receive auxiliary training; target encoder has only four training tasks",
            "Hash-based coding representation is bounded and loses token order",
        ],
    }
    return summary
