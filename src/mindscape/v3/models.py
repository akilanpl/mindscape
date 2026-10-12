"""Interchangeable prediction-only backends; no verifier/environment dependency."""

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from .schema import Action, Observation, Prediction, State, digest


class Backend(Protocol):
    identifier: str
    parameter_count: int

    def logits(self, observation: Observation, candidates: tuple[Action, ...]) -> list[float]: ...
    def predict_transition(self, state: State, action: Action, confidence: float) -> Prediction: ...
    def checkpoint(self) -> dict: ...


class TinyMLP:
    """Pure Python CPU tanh MLP, deliberately untrained for foundation controls.

    Same 7 -> 16 -> 65 policy backbone plus a 17 -> 3 transition head for all baseline conditions. No download required.
    Numerical predictions are not learned arithmetic claims until independently trained/tested.
    """

    identifier = "v3.tiny_mlp"
    policy_forward_calls = 1
    transition_forward_calls = 1
    input_size, hidden_size, output_size = 7, 16, 65
    parameter_count = 7 * 16 + 16 + 16 * 65 + 65 + 17 * 3 + 3

    def __init__(self, seed=0, zero=False):
        rng = random.Random(seed)
        self.seed = seed
        self.weights = [
            0.0 if zero else rng.uniform(-0.2, 0.2) for _ in range(self.parameter_count)
        ]
        self.condition = "zero" if zero else "random"

    @staticmethod
    def features(observation):
        domains = ("mathematics", "python", "dynamics")
        if len(observation.values) > 3 or observation.domain not in domains:
            raise ValueError("Unsupported feature schema")
        values = list(observation.values) + [0.0] * (3 - len(observation.values))
        return (
            [x / 32 for x in values]
            + [float(observation.domain == d) for d in domains]
            + [float(observation.partial)]
        )

    def logits(self, observation, candidates):
        import math

        if len(candidates) != self.output_size or tuple(a.value for a in candidates) != tuple(
            range(-32, 33)
        ):
            raise ValueError("Unexpected candidate vocabulary")
        f = self.features(observation)
        w = self.weights
        hidden = [
            math.tanh(sum(f[i] * w[h * 7 + i] for i in range(7)) + w[112 + h]) for h in range(16)
        ]
        return [
            sum(hidden[h] * w[128 + j * 16 + h] for h in range(16)) + w[1168 + j] for j in range(65)
        ]

    def predict_transition(self, state, action, confidence):
        # A separate untrained neural head predicts numerical next-state contents.
        # Execution does not consume these predictions and never corrects them.
        import math

        f = self.features(state.observation)
        w = self.weights
        hidden = [
            math.tanh(sum(f[i] * w[h * 7 + i] for i in range(7)) + w[112 + h]) for h in range(16)
        ] + [action.value / 32]
        size = 3 if state.observation.domain == "python" else 1
        predicted = tuple(
            32 * math.tanh(sum(hidden[h] * w[1233 + j * 17 + h] for h in range(17)) + w[1284 + j])
            for j in range(size)
        )
        return Prediction(action, State(state.observation, predicted, True), confidence)

    def checkpoint(self):
        return {
            "schema": "v3.tiny_mlp/1",
            "seed": self.seed,
            "condition": self.condition,
            "weights": self.weights,
            "parameter_count": self.parameter_count,
            "training_task_ids": [],
            "updates": 0,
            "transition_head": "untrained_neural_regression",
        }

    @property
    def checkpoint_hash(self):
        return digest(self.checkpoint())

    def save(self, path):
        Path(path).write_text(json.dumps(self.checkpoint(), allow_nan=False))

    @classmethod
    def load(cls, path, expected_hash):
        import math

        data = json.loads(Path(path).read_text())
        if digest(data) != expected_hash or data["schema"] != "v3.tiny_mlp/1":
            raise ValueError("Checkpoint hash/schema mismatch")
        model = cls(data["seed"], data["condition"] == "zero")
        if len(data["weights"]) != model.parameter_count or any(
            type(x) not in (float, int) or not math.isfinite(x) for x in data["weights"]
        ):
            raise ValueError("Invalid weights")
        model.weights = data["weights"]
        return model


class HistoricalBackend:
    """Optional local-only Transformers adapter. No implicit downloads or remote code."""

    policy_forward_calls = 65
    transition_forward_calls = 0
    approved_families = ("gpt_neo", "gpt_neox")  # GPT-Neo and Pythia configuration families

    def __init__(self, path, approved_hashes):
        import hashlib

        root = Path(path).resolve()
        if not root.is_dir() or not approved_hashes:
            raise ValueError("Approved local checkpoint manifest required")
        for name, expected in approved_hashes.items():
            file = (root / name).resolve()
            if not file.is_relative_to(root) or not file.is_file():
                raise ValueError("Unsafe checkpoint path")
            if hashlib.sha256(file.read_bytes()).hexdigest() != expected:
                raise ValueError("Historical checkpoint mismatch")
        actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
        if actual != set(approved_hashes):
            raise ValueError("Every local checkpoint file must be approved")
        config = json.loads((root / "config.json").read_text())
        if config.get("model_type") not in self.approved_families:
            raise ValueError("Unapproved model family")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            root, local_files_only=True, trust_remote_code=False
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            root, local_files_only=True, trust_remote_code=False, use_safetensors=True
        ).eval()
        self.identifier = "v3.local_historical"
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        self.manifest = dict(approved_hashes)

    def checkpoint(self):
        return {
            "identifier": self.identifier,
            "files": self.manifest,
            "pretraining_exposure": "unknown",
            "transition_head": "write_only_hypothesis",
        }

    def logits(self, observation, candidates):
        import torch

        prompt = json.dumps(asdict(observation), sort_keys=True) + "\nsubmit:"
        prefix = self.tokenizer.encode(prompt, add_special_tokens=False)
        scores = []
        with torch.no_grad():
            for action in candidates:
                suffix = self.tokenizer.encode(" " + str(action.value), add_special_tokens=False)
                ids = torch.tensor([prefix + suffix])
                logp = self.model(ids).logits.log_softmax(-1)
                scores.append(
                    sum(
                        float(logp[0, len(prefix) + j - 1, token]) for j, token in enumerate(suffix)
                    )
                )
        return scores

    def predict_transition(self, state, action, confidence):
        # Historical language checkpoints have no compatible learned transition head.
        # An explicit write-only hypothesis is retained and labelled in checkpoint metadata.
        return Prediction(
            action, State(state.observation, (float(action.value),), True), confidence
        )
