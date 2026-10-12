"""Offline frozen MiniLM representation plus separately trained residual adapter."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


class FrozenEncoder:
    def __init__(self, path, cache):
        from transformers import AutoModel, AutoTokenizer

        self.root = Path(path)
        if not (self.root / "model.safetensors").is_file():
            raise ValueError("Existing local safetensors required; downloads forbidden")
        self.hashes = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in self.root.iterdir()
            if p.is_file()
        }
        config = json.loads((self.root / "config.json").read_text())
        if config.get("model_type") != "bert":
            raise ValueError("Expected approved local BERT encoder")
        self.tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False
        )
        self.model = AutoModel.from_pretrained(
            path, local_files_only=True, trust_remote_code=False, use_safetensors=True
        ).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.size = self.model.config.hidden_size
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        self.cache_path = Path(cache)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.values = {}
        if self.cache_path.exists():
            with np.load(self.cache_path, allow_pickle=False) as data:
                self.values = {k: data[k] for k in data.files}
        self.tokens, self.calls, self.requests = 0, 0, 0

    @staticmethod
    def text(row):
        # Pure serialization; no arithmetic answers, split identities or verifier judgments.
        return "Observed numerical fields: " + " ".join(
            f"f{i}:{float(v):g}" for i, v in enumerate(row) if v
        )

    @torch.inference_mode()
    def encode(self, xs):
        xs = np.asarray(xs, dtype=np.float32)
        keys = [hashlib.sha256(row.tobytes()).hexdigest() for row in xs]
        todo = {k: x for k, x in zip(keys, xs) if k not in self.values}
        items = list(todo.items())
        for start in range(0, len(items), 32):
            batch = items[start : start + 32]
            inputs = self.tokenizer(
                [self.text(x) for _, x in batch],
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            if inputs["input_ids"].shape[1] > self.model.config.max_position_embeddings:
                raise ValueError("Prompt exceeds frozen encoder limit")
            self.calls += 1
            self.tokens += int(inputs["attention_mask"].sum())
            hidden = self.model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1)
            pooled = torch.nn.functional.normalize(pooled, dim=-1).numpy()
            for (key, _), value in zip(batch, pooled):
                self.values[key] = value
        self.requests += len(xs)
        return torch.from_numpy(np.stack([self.values[k] for k in keys]))

    def save(self):
        np.savez_compressed(self.cache_path, **self.values)
