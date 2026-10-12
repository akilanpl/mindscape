"""Batched MPS hypothetical rollouts; independent private targets used only afterward."""

import numpy as np
import torch

from .baselines import predict as baseline_predict
from .schema import DT, HISTORY


def public_batch(rows):
    return {
        "history": np.stack([r["observations"][:HISTORY] for r in rows]),
        "masks": np.stack([r["masks"][:HISTORY] for r in rows]),
        "actions": np.stack([r["actions"][HISTORY - 1 : HISTORY - 1 + r["horizon"]] for r in rows]),
        "contexts": np.stack(
            [r["contexts"][HISTORY - 1 : HISTORY - 1 + r["horizon"]] for r in rows]
        ),
    }


def rollout(model, public, kind=None):
    """Input contains no states/scenes/private parameters/engine handles."""
    if set(public) != {"history", "masks", "actions", "contexts"}:
        raise ValueError("Private prediction channel")
    history = public["history"].copy()
    masks = public["masks"].copy()
    predictions = []
    uncertainty = []
    contacts = []
    if model is not None:
        if kind is not None:
            raise ValueError("Mixed predictor")
        history = torch.as_tensor(history, device="mps", dtype=torch.float32)
        masks = torch.as_tensor(masks, device="mps", dtype=torch.float32)
        actions = torch.as_tensor(public["actions"], device="mps", dtype=torch.float32)
        contexts = torch.as_tensor(public["contexts"], device="mps", dtype=torch.float32)
        torch.mps.synchronize()
    for t in range(public["actions"].shape[1]):
        if model is not None:
            delta, var, contact = model.predict(history, masks, actions[:, t], contexts[:, t])
            current = history[:, -1].clone()
            current[:, :, :4] += delta
            current[:, :, 8] = contact
            predictions.append(current[:, :, :4])
            uncertainty.append(var)
            contacts.append(contact)
            history = torch.cat((history[:, 1:], current[:, None]), dim=1)
        else:
            predicted, contact = baseline_predict(
                history, masks, public["actions"][:, t], public["contexts"][:, t], kind
            )
            current = history[:, -1].copy()
            current[:, :, :4] = predicted
            current[:, :, 8] = contact
            predictions.append(predicted)
            uncertainty.append(np.zeros_like(predicted))
            contacts.append(contact)
            history = np.concatenate((history[:, 1:], current[:, None]), axis=1)
        # Predicted velocity is now a hypothetical estimate, no observed future supplied.
        masks[:, :-1] = masks[:, 1:].clone() if model is not None else masks[:, 1:].copy()
    if model is not None:
        torch.mps.synchronize()
        return {
            "states": torch.stack(predictions, 1).cpu().numpy(),
            "variance": torch.stack(uncertainty, 1).cpu().numpy(),
            "contacts": torch.stack(contacts, 1).cpu().numpy(),
        }
    return {
        "states": np.stack(predictions, 1),
        "variance": np.stack(uncertainty, 1),
        "contacts": np.stack(contacts, 1),
    }


def metrics(predictions, row, normalization):
    horizon = row["horizon"]
    truth = row["states"][HISTORY : HISTORY + horizon, :, :4]
    active = row["states"][HISTORY - 1, :, 9] > 0
    predicted = predictions["states"][:horizon]
    error = (predicted - truth)[:, active]
    if not np.isfinite(predicted).all():
        return {
            "finite": False,
            "normalized_rollout_mse": 1e6,
            "one_step_mse": 1e6,
            "goal": "prediction_only",
        }
    mse = float(np.mean(error**2))
    normalized = float(np.mean((error / normalization) ** 2))
    actual_contact = row["states"][HISTORY : HISTORY + horizon, :, 8][:, active]
    contact = predictions["contacts"][:horizon, active]
    variance = np.cumsum(predictions["variance"][:horizon, active], axis=0)
    positive = variance > 0
    coverage = (
        float(np.mean(abs(error[positive]) <= 1.96 * np.sqrt(variance[positive])))
        if positive.any()
        else None
    )
    radii = row["states"][HISTORY - 1, :, 5][active]
    floor_penetration = float(np.maximum(0, radii[None] - predicted[:, active, 1]).mean())
    mass = row["states"][HISTORY - 1, :, 4][active]
    momentum_error = float(
        np.mean(
            (
                np.sum(predicted[:, active, 2:4] * mass[None, :, None], axis=1)
                - np.sum(truth[:, active, 2:4] * mass[None, :, None], axis=1)
            )
            ** 2
        )
    )
    # Honest inference proxies, not supervised hidden-parameter estimates.
    dt_acc = (predicted[0, :, 2:4] - row["observations"][HISTORY - 1, :, 2:4]) / DT
    true_acc = (truth[0, :, 2:4] - row["states"][HISTORY - 1, :, 2:4]) / DT
    return {
        "finite": True,
        "one_step_mse": float(np.mean(error[0] ** 2)),
        "rollout_mse": mse,
        "normalized_rollout_mse": normalized,
        "final_position_rmse": float(np.sqrt(np.mean(error[-1, :, :2] ** 2))),
        "contact_accuracy": float(np.mean((contact >= 0.5) == (actual_contact >= 0.5))),
        "contact_brier": float(np.mean((contact - actual_contact) ** 2)),
        "uncertainty_coverage95": coverage,
        "floor_penetration_mean": floor_penetration,
        "momentum_total_mse": momentum_error,
        "acceleration_proxy_mse": float(np.mean((dt_acc[active] - true_acc[active]) ** 2)),
        "calibration_limit": "sum of per-step diagonal variances, temporal correlation ignored; not epistemic uncertainty",
    }
