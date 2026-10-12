"""Explicit nonlearned equation baseline; public parameters or declared nominal defaults."""

import argparse
from pathlib import Path

import numpy as np

from mindscape.v3.physics_world import evaluation
from mindscape.v3.physics_world.baselines import predict as free_flight
from mindscape.v3.physics_world.data import load
from mindscape.v3.physics_world.storage import RunLock, atomic_json, source_hash
from mindscape.v3.physics_world.study import evaluate, normalization


def predict(history, masks, action, context, kind):
    current = history[:, -1]
    mass = np.where(masks[:, -1, :, 4] > 0, current[:, :, 4], 1.0)
    mass = np.maximum(mass, 0.05)
    modified = action.copy()
    # Fixed known rest length1.5, known spring coefficient. Approximate free-flight integration.
    for batch in range(len(current)):
        if context[batch, 2] and current[batch, 1, 9]:
            delta = current[batch, 1, :2] - current[batch, 0, :2]
            distance = np.linalg.norm(delta)
            force = context[batch, 2] * (distance - 1.5) * delta / max(distance, 1e-6)
            modified[batch, 0, :2] += force
            modified[batch, 1, :2] -= force
    result, contact = free_flight(history, masks, modified, context, "analytical_available")
    for batch in range(len(current)):
        for i in range(3):
            if not current[batch, i, 9]:
                continue
            radius = current[batch, i, 5]
            restitution = current[batch, i, 7] if masks[batch, -1, i, 7] else 0.5
            friction = current[batch, i, 6] if masks[batch, -1, i, 6] else 0.2
            if result[batch, i, 1] < radius + 0.02:
                result[batch, i, 1] = radius + 0.02
                if result[batch, i, 3] < 0:
                    result[batch, i, 3] *= -restitution
                reduction = friction * abs(context[batch, 0]) / 30
                result[batch, i, 2] = np.sign(result[batch, i, 2]) * max(
                    0, abs(result[batch, i, 2]) - reduction
                )
                contact[batch, i] = 1
            for j in range(i + 1, 3):
                if not current[batch, j, 9]:
                    continue
                delta = result[batch, j, :2] - result[batch, i, :2]
                distance = np.linalg.norm(delta)
                normal = delta / max(distance, 1e-6)
                closing = np.dot(result[batch, j, 2:4] - result[batch, i, 2:4], normal)
                if distance < current[batch, i, 5] + current[batch, j, 5] and closing < 0:
                    e2 = current[batch, j, 7] if masks[batch, -1, j, 7] else 0.5
                    impulse = (
                        -(1 + restitution * e2)
                        * closing
                        / (1 / mass[batch, i] + 1 / mass[batch, j])
                    )
                    result[batch, i, 2:4] -= impulse / mass[batch, i] * normal
                    result[batch, j, 2:4] += impulse / mass[batch, j] * normal
                    contact[batch, [i, j]] = 1
    return result, contact


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    manifest = load(a.dataset)
    identity = {
        "dataset": manifest["hash"],
        "source": source_hash(),
        "device": "cpu_non_neural",
        "purpose": "declared_contact_spring_equations_no_simulator_queries",
    }
    lock = RunLock(a.output, identity)
    evaluation.baseline_predict = predict
    try:
        for family in ("motion", "interaction"):
            scale = normalization(a.dataset, family)
            for split in ("iid", "ood"):
                records = [
                    r
                    for r in manifest["records"]
                    if r["scene"]["family"] == family and r["scene"]["split"] == split
                ]
                evaluate(
                    a.dataset,
                    records,
                    None,
                    "analytical_contacts",
                    Path(a.output) / f"{family}/analytical_contacts/evaluation/{split}",
                    {**identity, "family": family, "variant": "analytical_contacts"},
                    scale,
                )
        atomic_json(
            Path(a.output) / "receipt.json",
            {
                "nonneural_rollouts": 400,
                "neural_calls": 0,
                "engine_prediction_queries": 0,
                "defaults": "Hidden mass1, restitution0.5, friction0.2; known fixed floor/rest length. Approximate integration, not oracle.",
            },
        )
    finally:
        lock.close()
