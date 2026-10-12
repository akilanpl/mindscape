"""New frozen training/development scenes and independently seeded sealed scenes."""

from dataclasses import replace
from pathlib import Path

import numpy as np

from mindscape.v3.physics_world.data import generate_scene, load
from mindscape.v3.physics_world.environment import ENGINE_REVISION, ENGINE_VERSION, trajectory
from mindscape.v3.physics_world.schema import HISTORY, digest
from mindscape.v3.physics_world.storage import atomic_json, file_sha


def create(root, protocol):
    root = Path(root)
    if root.exists():
        return load(root)
    root.mkdir(parents=True)
    records = []
    for family in ("motion", "interaction"):
        panels = [
            ("train", 25, False, False),
            ("development_iid", 6, False, False),
            ("development_stress", 7, True, False),
            ("sealed_iid", 12, False, True),
            ("sealed_ood", 14, True, True),
        ]
        for panel, count, stress, sealed in panels:
            seed = protocol["sealed_seed"] if sealed else protocol["training_seed"]
            for i in range(count):
                panel_seed = int(digest([seed, panel])[:8], 16)
                source = generate_scene(family, "ood" if stress else "train", i, panel_seed)
                # New multibody training exposure, declared before any V3.1 result.
                if family == "motion" and not stress and i % 2:
                    neighbor = replace(source.objects[0], position=(1.4, 2.0), velocity=(-1.5, 1.0))
                    source = replace(source, objects=(*source.objects, neighbor))
                if stress and source.ood_condition == "configuration" and family == "motion":
                    neighbor = replace(source.objects[0], position=(2.5, 2.2), velocity=(-1.0, 0.5))
                    source = replace(source, objects=(*source.objects, neighbor))
                scene = replace(source, split=panel, index=i, horizon=32)
                row = trajectory(scene)
                file = root / f"{scene.identity}.npz"
                np.savez_compressed(file, **{k: v for k, v in row.items() if k != "identity"})
                records.append(
                    {
                        "identity": scene.identity,
                        "scene": scene.record(),
                        "file": file.name,
                        "sha256": file_sha(file),
                    }
                )
    assert len({r["identity"] for r in records}) == len(records)
    manifest = {
        "schema": "mindscape.v3.1.physics.dataset/1",
        "engine": ENGINE_VERSION,
        "engine_revision": ENGINE_REVISION,
        "records": records,
        "protocol": digest(protocol),
        "history": HISTORY,
        "sealed_access": "pilot must not open sealed trajectory files",
    }
    manifest["hash"] = digest(manifest)
    atomic_json(root / "manifest.json", manifest)
    return manifest


def windows(root, family, length=8):
    manifest = load(root)
    rows = []
    ids = []
    for record in manifest["records"]:
        if record["scene"]["family"] != family or record["scene"]["split"] != "train":
            continue
        ids.append(record["identity"])
        with np.load(Path(root) / record["file"], allow_pickle=False) as d:
            for t in range(HISTORY - 1, len(d["actions"]) - length + 1):
                rows.append(
                    (
                        d["observations"][t - HISTORY + 1 : t + 1],
                        d["masks"][t - HISTORY + 1 : t + 1],
                        d["actions"][t : t + length],
                        d["contexts"][t : t + length],
                        d["states"][t + 1 : t + length + 1, :, :4],
                        d["states"][t + 1 : t + length + 1, :, 8],
                    )
                )
    keys = ("history", "masks", "actions", "contexts", "targets", "contacts")
    return {**{k: np.asarray([row[i] for row in rows]) for i, k in enumerate(keys)}, "ids": ids}
