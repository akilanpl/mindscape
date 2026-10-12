"""CPU nonneural predictions, using only allowed observations and context."""

import numpy as np

from .schema import DT


def predict(history, masks, action, context, kind):
    current = history[:, -1].copy()
    active = current[:, :, 9:10]
    velocity = np.where(
        masks[:, -1, :, 2:4] > 0,
        current[:, :, 2:4],
        (history[:, -1, :, :2] - history[:, -2, :, :2]) / DT,
    )
    if kind == "persistence":
        return current[:, :, :4], np.zeros(current.shape[:2])
    next_state = current[:, :, :4].copy()
    if kind == "constant_velocity":
        next_state[:, :, :2] += DT * velocity
        next_state[:, :, 2:4] = velocity
    elif kind == "analytical_available":
        mass = np.where(masks[:, -1, :, 4] > 0, current[:, :, 4], 1.0).clip(0.05, None)
        acceleration = action[:, :, :2] / mass[:, :, None]
        acceleration[:, :, 1] -= context[:, None, 0]
        velocity = velocity + action[:, :, 2:] / mass[:, :, None]
        # Free-flight equation; contact, friction and spring physics not secretly queried.
        next_state[:, :, :2] += DT * velocity + 0.5 * DT**2 * acceleration
        next_state[:, :, 2:4] = velocity + DT * acceleration
    else:
        raise ValueError("Unknown baseline")
    return next_state * active, np.zeros(current.shape[:2])
