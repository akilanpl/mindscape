"""Seeded mini-batch Adam with validation-loss checkpoint selection."""
import numpy as np


def fit(backend, train_x, train_y, val_x, val_y, steps=400, batch_size=64,
        learning_rate=0.003, seed=0, validation_every=50):
    rng = np.random.default_rng(seed)
    moments = {k: np.zeros_like(v) for k, v in backend.weights.items()}
    variances = {k: np.zeros_like(v) for k, v in backend.weights.items()}
    best, best_step, best_weights, logs = float("inf"), 0, None, []
    initial_val, _ = backend.loss_and_gradients(val_x, val_y)
    logs.append({"step": 0, "validation_loss": initial_val})
    for step in range(1, steps + 1):
        indices = rng.integers(0, len(train_x), size=batch_size)
        loss, gradients = backend.loss_and_gradients(train_x[indices], train_y[indices])
        for key, gradient in gradients.items():
            moments[key] = .9 * moments[key] + .1 * gradient
            variances[key] = .999 * variances[key] + .001 * gradient ** 2
            backend.weights[key] -= learning_rate * (moments[key] / (1 - .9 ** step)) / (
                np.sqrt(variances[key] / (1 - .999 ** step)) + 1e-8)
        if step % validation_every == 0 or step == steps:
            val_loss, _ = backend.loss_and_gradients(val_x, val_y)
            logs.append({"step": step, "training_batch_loss": loss, "validation_loss": val_loss})
            if val_loss < best:
                best, best_step = val_loss, step
                best_weights = {k: v.copy() for k, v in backend.weights.items()}
    backend.weights = best_weights
    predictions = backend.generate(val_x)
    return logs, {"best_validation_loss": best, "best_step": best_step,
                  "validation_label_accuracy": float(np.mean(predictions == val_y)),
                  "validation_exact_row_accuracy": float(np.mean(np.all(predictions == val_y, axis=1))),
                  "initial_validation_loss": initial_val}
