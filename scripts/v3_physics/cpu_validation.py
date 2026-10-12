"""Nonneural engine/baseline evidence only; no CPU neural computation."""

import argparse
import json
from pathlib import Path

import numpy as np

from mindscape.v3.physics_world.data import load
from mindscape.v3.physics_world.environment import trajectory
from mindscape.v3.physics_world.schema import Scene
from mindscape.v3.physics_world.storage import RunLock, atomic_json, source_hash
from mindscape.v3.physics_world.study import evaluate, normalization

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = load(args.dataset)
    identity = {
        "dataset": manifest["hash"],
        "source": source_hash(),
        "device": "cpu_non_neural",
        "purpose": "independent_simulation_and_baseline_preflight",
    }
    root = Path(args.output)
    lock = RunLock(root, identity)
    verified = 0
    try:
        for record in manifest["records"]:
            simulated = trajectory(Scene.from_record(record["scene"]))
            with np.load(Path(args.dataset) / record["file"], allow_pickle=False) as stored:
                for key in ("states", "actions", "observations", "masks", "contexts"):
                    np.testing.assert_array_equal(simulated[key], stored[key])
            verified += 1
        for family in ("motion", "interaction"):
            scale = normalization(args.dataset, family)
            for kind in ("persistence", "constant_velocity", "analytical_available"):
                for split in ("iid", "ood"):
                    records = [
                        r
                        for r in manifest["records"]
                        if r["scene"]["family"] == family and r["scene"]["split"] == split
                    ]
                    evaluate(
                        args.dataset,
                        records,
                        None,
                        kind,
                        root / f"{family}/{kind}/evaluation/{split}",
                        {**identity, "family": family, "variant": kind},
                        scale,
                    )
        atomic_json(
            root / "receipt.json",
            {
                "simulation_scenes_replayed": verified,
                "nonneural_prediction_rollouts": 1200,
                "neural_calls": 0,
                "device": "cpu_non_neural",
                "scope": "Preflight evidence; separate from full study baselines and never primary neural episodes",
            },
        )
        print(json.dumps({"replayed_scenes": verified, "neural_calls": 0}))
    finally:
        lock.close()
