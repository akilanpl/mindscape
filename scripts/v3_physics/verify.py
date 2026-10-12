"""Independent CPU simulator replay, metric recomputation, integrity/coverage checks."""

import argparse
import json
from pathlib import Path

import numpy as np

from mindscape.v3.physics_world.data import load
from mindscape.v3.physics_world.environment import Environment, trajectory
from mindscape.v3.physics_world.evaluation import metrics
from mindscape.v3.physics_world.schema import HISTORY, Action, Scene, digest
from mindscape.v3.physics_world.storage import atomic_json, file_sha


def verify(dataset, root, output):
    manifest = load(dataset)
    by_id = {r["identity"]: r for r in manifest["records"]}
    replayed = {}
    count = 0
    planned = 0
    checkpoints = 0
    for pointer in Path(root).rglob("checkpoint-index.json"):
        entry = json.loads(pointer.read_text())
        if (
            Path(entry["file"]).name != entry["file"]
            or file_sha(pointer.parent / entry["file"]) != entry["sha256"]
        ):
            raise ValueError("Checkpoint corruption")
        checkpoints += 1
    for path in Path(root).rglob("*.json"):
        if len(path.stem) != 64:
            continue
        saved = json.loads(path.read_text())
        trace = saved["trace"]
        if saved["trace_hash"] != digest(trace):
            raise ValueError("Raw evidence hash")
        key = trace["scene_id"]
        record = by_id[key]
        if "metrics" not in trace:
            scene = Scene.from_record(record["scene"])
            env = Environment(scene)
            for _ in range(HISTORY - 1):
                env.step(Action.from_vector(np.zeros((3, 4))))
            touched = False
            for t in range(24):
                actual = env.step(
                    Action.from_vector(trace["proposal"]["action"] if t == 0 else np.zeros((3, 4)))
                )
                touched |= bool(actual[0, 8])
            np.testing.assert_array_equal(actual, np.asarray(trace["actual_terminal"]))
            distance = float(np.linalg.norm(actual[0, :2] - np.asarray(trace["target"])))
            if (
                not np.isclose(distance, trace["distance"])
                or (distance <= 0.75) != trace["goal_success"]
                or touched != trace["floor_or_object_contact"]
            ):
                raise ValueError("Actual planning outcome changed")
            planned += 1
            continue
        if key not in replayed:
            simulated = trajectory(Scene.from_record(record["scene"]))
            with np.load(Path(dataset) / record["file"], allow_pickle=False) as original:
                for field in ("states", "actions", "observations", "masks", "contexts"):
                    np.testing.assert_array_equal(simulated[field], original[field])
            simulated["horizon"] = record["scene"]["horizon"]
            replayed[key] = simulated
        prediction = {k: np.asarray(v, dtype=np.float32) for k, v in trace["predictions"].items()}
        recomputed = metrics(prediction, replayed[key], np.asarray(trace["normalization"]))
        for field, value in recomputed.items():
            expected = trace["metrics"][field]
            if isinstance(value, (float, int)):
                if not np.isclose(value, expected, rtol=1e-5, atol=1e-6):
                    raise ValueError(f"Metric changed:{field}")
            elif value != expected:
                raise ValueError("Metric label changed")
        if trace["device"] == "mps" and saved["identity"]["device"] != "mps":
            raise ValueError("Device provenance")
        count += 1
    receipt = {
        "recomputed_prediction_episodes": count,
        "independent_simulation_scenes": len(replayed),
        "neural_calls": 0,
        "dataset_hash": manifest["hash"],
        "planning_episodes_independently_replayed": planned,
        "checkpoint_indices_verified": checkpoints,
        "planning_verification": "actual selected actions replayed in independent simulator; no neural queries",
    }
    atomic_json(output, receipt)
    print(json.dumps(receipt))
    return receipt


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    verify(a.dataset, a.run, a.output)
