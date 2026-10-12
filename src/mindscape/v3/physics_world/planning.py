"""Bounded batched learned trajectory ranking; no simulator import or query."""

import numpy as np

from .evaluation import rollout
from .schema import MAX_OBJECTS, digest

CANDIDATES = np.asarray(
    [(x, y) for x in (-2.0, 0.0, 2.0) for y in (0.0, 1.0, 2.0)], dtype=np.float32
)


def plan(model, observation_history, masks, context, target, kind="learned", seed=11, horizon=24):
    # All nine candidates remain plausible; no exact semantic mask or target action supplied.
    actions = np.zeros((len(CANDIDATES), horizon, MAX_OBJECTS, 4), dtype=np.float32)
    actions[:, 0, 0, 2:] = CANDIDATES
    contexts = np.tile(context, (len(CANDIDATES), horizon, 1))
    contexts[:, :, 3] += np.arange(horizon)[None] / 30
    public = {
        "history": np.repeat(observation_history[None], len(CANDIDATES), 0),
        "masks": np.repeat(masks[None], len(CANDIDATES), 0),
        "actions": actions,
        "contexts": contexts,
    }
    if kind == "random":
        index = int(np.random.default_rng(seed).integers(len(CANDIDATES)))
        predicted = None
    elif kind == "simple":
        index = int(
            np.argmin(
                np.sum(
                    (CANDIDATES - (np.asarray(target) - observation_history[-1, 0, :2])) ** 2,
                    axis=1,
                )
            )
        )
        predicted = None
    else:
        result = rollout(
            model, public, kind="analytical_available" if kind == "analytical" else None
        )
        terminal = result["states"][:, -1, 0, :2]
        distance = np.sum((terminal - target) ** 2, axis=1)
        risk = np.sum(result["variance"][:, -1, 0, :2], axis=1)
        # Hypothetical floor penalty; actual goal/obstacle outcome independently checked later.
        collision = np.maximum(0, 0.25 - result["states"][:, :, 0, 1]).mean(axis=1)
        scores = distance + 0.05 * risk + 2 * collision
        index = int(np.argmin(scores))
        predicted = {
            "states": result["states"][index].tolist(),
            "scores": scores.tolist(),
            "kind": "hypothetical_unverified",
        }
    return {
        "selected": index,
        "action": actions[index, 0].tolist(),
        "candidate_impulses": CANDIDATES.tolist(),
        "hypothesis": predicted,
        "objective": "terminal squared target distance +.05 variance +2 predicted floor penetration",
        "model_candidate_rollouts": len(CANDIDATES) if kind in ("learned", "analytical") else 0,
        "environment_queries": 0,
        "trace_hash": digest([index, actions[index, 0].tolist(), target.tolist()]),
    }
