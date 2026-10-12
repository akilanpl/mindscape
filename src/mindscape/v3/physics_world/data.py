"""Immutable scene-level splits and CPU simulation; budgets count trajectories."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .environment import ENGINE_REVISION, ENGINE_VERSION, trajectory
from .schema import HISTORY, ObjectSpec, Scene, digest

BUDGETS = (10, 25, 50, 100, 250)
SEEDS = (11, 23, 37)
OOD = (
    "mass_friction",
    "velocity",
    "horizon",
    "configuration",
    "noise",
    "parameters",
    "interaction",
)


def generate_scene(family, split, index, seed=20261012):
    if family not in ("motion", "interaction") or split not in (
        "train",
        "validation",
        "iid",
        "ood",
    ):
        raise ValueError("Unknown split/family")
    rng = np.random.default_rng(int(digest([seed, family, split, index])[:16], 16))
    ood = OOD[index % len(OOD)] if split == "ood" else "none"
    mass = float(rng.uniform(0.8, 1.8))
    friction = float(rng.uniform(0.1, 0.4))
    speed = float(rng.uniform(1, 3))
    gravity = float(rng.uniform(8, 10)) if family == "motion" else 0.0
    count = 1 if family == "motion" else 2
    if ood == "mass_friction":
        mass = float(rng.uniform(2.5, 4))
        friction = float(rng.uniform(0.65, 0.9))
    if ood == "velocity":
        speed = float(rng.uniform(4, 6))
    if ood == "configuration":
        count += 1
    if ood == "parameters":
        gravity = 14.0 if family == "motion" else 0.8
    spring = float(rng.uniform(2, 6)) if family == "interaction" and index % 2 else 0.0
    if ood == "interaction":
        spring = 9.0 if family == "interaction" else 0.0
        count = max(2, count)
    objects = []
    for i in range(count):
        if family == "motion":
            position = (-1 + i * 1.2, float(rng.uniform(1.5, 3)))
            velocity = (speed * (-1 if i else 1), float(rng.uniform(1, 3)))
        else:
            position = (-1 + i * 1.8, 0.45 if not spring else 1.5)
            velocity = (speed * (-1 if i else 1), 0.0)
        ratio = float(rng.uniform(0.8, 1.2)) if i else 1.0
        if ood == "mass_friction" and i:
            ratio = 0.25
        objects.append(
            ObjectSpec(
                position, velocity, mass * ratio, 0.25, friction, float(rng.uniform(0.4, 0.95))
            )
        )
    return Scene(
        family,
        split,
        index,
        seed,
        "spring" if spring else ("projectile" if family == "motion" else "collision"),
        tuple(objects),
        gravity,
        0.06 if ood == "noise" else 0.005,
        bool(index % 3 == 0),
        bool(index % 5 == 0),
        spring,
        64 if ood == "horizon" else 32,
        ood,
    )


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(root, counts=None, workers=2, seed=20261012):
    root = Path(root)
    if root.exists():
        raise ValueError("Dataset destination must be new")
    counts = counts or {"train": 250, "validation": 50, "iid": 100, "ood": 100}
    if set(counts) != {"train", "validation", "iid", "ood"}:
        raise ValueError("All splits required")
    scenes = [
        generate_scene(family, split, i, seed=seed)
        for family in ("motion", "interaction")
        for split, n in counts.items()
        for i in range(n)
    ]
    if len({s.identity for s in scenes}) != len(scenes):
        raise ValueError("Identity leakage")
    root.mkdir(parents=True)
    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for scene, row in zip(scenes, pool.map(trajectory, scenes), strict=True):
            path = root / (scene.identity + ".npz")
            np.savez_compressed(path, **{k: v for k, v in row.items() if k != "identity"})
            records.append(
                {
                    "scene": scene.record(),
                    "identity": scene.identity,
                    "file": path.name,
                    "sha256": sha(path),
                    "frames": len(row["states"]),
                    "finite": bool(np.isfinite(row["states"]).all()),
                }
            )
    manifest = {
        "schema": "mindscape.v3.physics.dataset/1",
        "engine": ENGINE_VERSION,
        "engine_revision": ENGINE_REVISION,
        "history": HISTORY,
        "counts_per_family": counts,
        "budget_unit": "independent trajectory, not frame",
        "public_boundary": "observations,masks,actions,contexts only; states and scenes are private simulator/scorer records",
        "records": records,
        "scope": "full_fixed"
        if counts == {"train": 250, "validation": 50, "iid": 100, "ood": 100}
        else "pilot_software_only",
    }
    manifest["hash"] = digest(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load(root, verify=True):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    saved = manifest.pop("hash")
    if digest(manifest) != saved:
        raise ValueError("Manifest changed")
    manifest["hash"] = saved
    if verify:
        seen = set()
        for row in manifest["records"]:
            scene = Scene.from_record(row["scene"])
            if scene.identity != row["identity"] or scene.identity in seen:
                raise ValueError("Identity leakage/tampering")
            seen.add(scene.identity)
            if Path(row["file"]).name != row["file"] or sha(root / row["file"]) != row["sha256"]:
                raise ValueError("Dataset corruption/path")
    return manifest


def training_arrays(root, family, budget):
    manifest = load(root)
    records = [
        r
        for r in manifest["records"]
        if r["scene"]["family"] == family and r["scene"]["split"] == "train"
    ][:budget]
    if len(records) != budget:
        raise ValueError("Insufficient trajectories")
    histories = []
    masks = []
    actions = []
    contexts = []
    targets = []
    contacts = []
    for record in records:
        with np.load(Path(root) / record["file"], allow_pickle=False) as data:
            for t in range(HISTORY - 1, len(data["actions"])):
                histories.append(data["observations"][t - HISTORY + 1 : t + 1])
                masks.append(data["masks"][t - HISTORY + 1 : t + 1])
                actions.append(data["actions"][t])
                contexts.append(data["contexts"][t])
                targets.append(data["states"][t + 1, :, :4] - data["observations"][t, :, :4])
                contacts.append(data["states"][t + 1, :, 8])
    return {
        "history": np.asarray(histories),
        "masks": np.asarray(masks),
        "actions": np.asarray(actions),
        "context": np.asarray(contexts),
        "delta": np.asarray(targets),
        "contact": np.asarray(contacts),
        "trajectory_ids": [r["identity"] for r in records],
    }
