"""Nonneural descriptive metrics; candidate probabilities are not task success probabilities."""

import numpy as np


def metrics(rows, predictions):
    labels, probs, correct, solvable, conditional = [], [], [], [], []
    for row, prediction in zip(rows, predictions):
        if prediction["identity"] != row["identity"]:
            raise ValueError("Prediction/task identity mismatch")
        y = row["labels"]
        success = y[prediction["selected"]]
        correct.append(success)
        solvable.append(float(any(y)))
        if any(y):
            conditional.append(success)
        if prediction.get("probabilities") is not None:
            labels.extend(y)
            probs.extend(prediction["probabilities"])
    result = {
        "tasks": len(rows),
        "correct": int(sum(correct)),
        "accuracy": float(np.mean(correct)),
        "solvable": int(sum(solvable)),
        "oracle_accuracy": float(np.mean(solvable)),
        "conditional_accuracy": float(np.mean(conditional)) if conditional else None,
    }
    if probs:
        y, p = np.asarray(labels), np.asarray(probs)
        ece = 0.0
        for i in range(5):
            mask = (p >= i / 5) & (p < (i + 1) / 5 if i < 4 else p <= 1)
            if mask.any():
                ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
        result.update(brier=float(((y - p) ** 2).mean()), ece_5_bins=float(ece))
    return result
