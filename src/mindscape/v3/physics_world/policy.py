"""MPS model-free policy; approved training-only executed action labels, no private eval access."""

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .accelerator import require_mps, verify_placement
from .environment import Environment
from .planning import CANDIDATES
from .schema import HISTORY, Action, Scene, observe
from .storage import atomic_json, file_sha
from .training import checkpoint, restore


def teacher_data(manifest, family, root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "policy-training-labels.json"
    approved = [
        r
        for r in manifest["records"]
        if r["scene"]["family"] == family and r["scene"]["split"] == "train"
    ]
    if path.exists():
        saved = json.loads(path.read_text())
        if (
            saved["training_ids"] != [r["identity"] for r in approved]
            or saved["dataset"] != manifest["hash"]
        ):
            raise ValueError("Policy approval changed")
        return saved
    rows = []
    steps = 0
    for record in approved:
        scene = Scene.from_record(record["scene"])
        env = Environment(scene)
        history = []
        context = None
        for t in range(HISTORY):
            observation = observe(env.state(), scene, t)
            history.append(observation["values"])
            context = observation["context"]
            if t < HISTORY - 1:
                env.step(Action.from_vector(np.zeros((3, 4))))
        target = np.asarray([history[-1][0, 0] + 1, max(0.6, history[-1][0, 1])], dtype=np.float32)
        distances = []
        for impulse in CANDIDATES:
            probe = Environment(scene)
            for _ in range(HISTORY - 1):
                probe.step(Action.from_vector(np.zeros((3, 4))))
            vector = np.zeros((3, 4))
            vector[0, 2:] = impulse
            for t in range(24):
                state = probe.step(Action.from_vector(vector if t == 0 else np.zeros((3, 4))))
            distances.append(float(np.linalg.norm(state[0, :2] - target)))
            steps += HISTORY - 1 + 24
        rows.append(
            {
                "scene_id": scene.identity,
                "input": np.concatenate((np.asarray(history).flatten(), context, target)).tolist(),
                "label": int(np.argmin(distances)),
                "training_candidate_distances": distances,
            }
        )
    saved = {
        "dataset": manifest["hash"],
        "training_ids": [r["identity"] for r in approved],
        "rows": rows,
        "actual_teacher_environment_steps": steps,
        "information_limit": "Additional executed action-choice supervision; not equal-information dynamics control",
    }
    atomic_json(path, saved)
    return saved


class ModelFreePolicy(nn.Module):
    def __init__(self, seed):
        device = require_mps()
        torch.manual_seed(seed)
        with torch.device(device):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(126, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 9)
            )
            self.register_buffer("scale", torch.ones(126, device=device))
        self.memory_keys = None
        self.memory_residuals = None
        self.memory_active = None

    def forward(self, x):
        verify_placement(self, x)
        return self.network(x / self.scale)

    def choose(self, history, context, target):
        x = np.concatenate((history.flatten(), context, target))
        with torch.inference_mode():
            return int(self(torch.as_tensor(x, device="mps", dtype=torch.float32)).argmax().item())


def fit_policy(data, output, identity, updates=400):
    require_mps()
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    model = ModelFreePolicy(identity["seed"])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    x = torch.as_tensor(
        np.asarray([r["input"] for r in data["rows"]]), device="mps", dtype=torch.float32
    )
    labels = torch.as_tensor([r["label"] for r in data["rows"]], device="mps")
    path = root / "checkpoint-index.json"
    step = restore(path, model, optimizer, identity) if path.exists() else 0
    if not path.exists():
        model.scale.copy_(x.std(0).clamp(min=0.1))
    initial = step
    started = time.perf_counter()
    for update in range(step, updates):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(x), labels)
        loss.backward()
        if update == initial:
            verify_placement(model, x, gradients=True)
        optimizer.step()
        step = update + 1
        if step % 50 == 0 or step == updates:
            torch.mps.synchronize()
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("Nonfinite policy loss")
            checkpoint(path, model, optimizer, step, identity)
    receipt = {
        "identity": identity,
        "step": step,
        "new_updates": step - initial,
        "device": "mps",
        "training_scenes": len(data["rows"]),
        "teacher_environment_steps": data["actual_teacher_environment_steps"],
        "checkpoint": file_sha(path),
        "seconds": time.perf_counter() - started,
        "information_limit": data["information_limit"],
    }
    atomic_json(root / f"training-invocation-{time.time_ns()}.json", receipt)
    return model, receipt
