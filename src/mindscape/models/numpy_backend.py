"""Small trainable task MLP, not a foundation language model."""
import json
from pathlib import Path
import numpy as np

from mindscape.models.encoding import INPUT_SIZE


class NumpyMLP:
    identifier = "numpy_tanh_mlp_v1"

    def __init__(self, groups, hidden=64, seed=0):
        self.groups = list(groups)
        self.hidden = hidden
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.weights = {
            "w1": rng.normal(0, np.sqrt(1 / INPUT_SIZE), (INPUT_SIZE, hidden)),
            "b1": np.zeros(hidden),
            "w2": rng.normal(0, np.sqrt(1 / hidden), (hidden, sum(groups))),
            "b2": np.zeros(sum(groups)),
        }

    @property
    def parameter_count(self):
        return sum(value.size for value in self.weights.values())

    def logits(self, x):
        h = np.tanh(np.asarray(x) @ self.weights["w1"] + self.weights["b1"])
        return h @ self.weights["w2"] + self.weights["b2"]

    def generate(self, x):
        return np.stack([np.argmax(group, axis=-1) for group in
                         np.split(self.logits(x), np.cumsum(self.groups)[:-1], axis=-1)], axis=-1)

    def loss_and_gradients(self, x, labels):
        w = self.weights
        h = np.tanh(x @ w["w1"] + w["b1"])
        logits = h @ w["w2"] + w["b2"]
        grads, loss, offset = [], 0.0, 0
        for index, size in enumerate(self.groups):
            scores = logits[:, offset:offset + size]
            shifted = scores - scores.max(axis=1, keepdims=True)
            exp = np.exp(shifted)
            probs = exp / exp.sum(axis=1, keepdims=True)
            target = labels[:, index]
            loss += float(np.mean(-shifted[np.arange(len(x)), target] + np.log(exp.sum(axis=1)))) / len(self.groups)
            probs[np.arange(len(x)), target] -= 1
            grads.append(probs / (len(x) * len(self.groups)))
            offset += size
        dz = np.concatenate(grads, axis=1)
        dh = (dz @ w["w2"].T) * (1 - h * h)
        return loss, {"w2": h.T @ dz, "b2": dz.sum(axis=0),
                      "w1": x.T @ dh, "b1": dh.sum(axis=0)}

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(path / "weights.npz", **self.weights)
        (path / "backend.json").write_text(json.dumps({"backend": self.identifier,
            "groups": self.groups, "hidden": self.hidden, "seed": self.seed,
            "parameter_count": self.parameter_count}, indent=2))

    @classmethod
    def load(cls, path):
        path = Path(path)
        metadata = json.loads((path / "backend.json").read_text())
        if metadata["backend"] != cls.identifier:
            raise ValueError("Unknown backend checkpoint")
        model = cls(metadata["groups"], metadata["hidden"], metadata["seed"])
        with np.load(path / "weights.npz", allow_pickle=False) as arrays:
            for key, value in model.weights.items():
                if arrays[key].shape != value.shape or not np.isfinite(arrays[key]).all():
                    raise ValueError("Invalid checkpoint weights")
                model.weights[key] = arrays[key].copy()
        return model
