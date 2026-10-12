"""Independent replay and statistics from raw development traces; no neural calls."""

import json
from pathlib import Path

import numpy as np

from mindscape.v3.physics_world.data import load
from mindscape.v3.physics_world.environment import Environment, trajectory
from mindscape.v3.physics_world.evaluation import metrics
from mindscape.v3.physics_world.schema import Action, Scene, digest
from mindscape.v3.physics_world.storage import atomic_json, file_sha


def verify(root):
    root = Path(root)
    dataset = Path("results/v3_1_physics/dataset-pilot-v1")
    manifest = load(dataset)
    lookup = {r["identity"]: r for r in manifest["records"]}
    rows = {}
    plans = 0
    checkpoints = 0
    groups = {}
    for path in root.rglob("checkpoint-index.json"):
        d = json.loads(path.read_text())
        assert Path(d["file"]).name == d["file"]
        assert file_sha(path.parent / d["file"]) == d["sha256"]
        checkpoints += 1
    for path in root.rglob("*.json"):
        if len(path.stem) != 64:
            continue
        saved = json.loads(path.read_text())
        t = saved["trace"]
        i = saved["identity"]
        assert digest(t) == saved["trace_hash"]
        record = lookup[t["scene_id"]]
        assert record["scene"]["split"].startswith("development")
        if "metrics" not in t:
            env = Environment(Scene.from_record(record["scene"]))
            for _ in range(3):
                env.step(Action.from_vector(np.zeros((3, 4))))
            for step in range(24):
                actual = env.step(
                    Action.from_vector(t["proposal"]["action"] if step == 0 else np.zeros((3, 4)))
                )
            np.testing.assert_array_equal(actual, t["actual_terminal"])
            distance = float(np.linalg.norm(actual[0, :2] - t["target"]))
            assert np.isclose(distance, t["distance"]) and (distance <= 0.75) == t["goal_success"]
            plans += 1
            continue
        if t["scene_id"] not in rows:
            simulated = trajectory(Scene.from_record(record["scene"]))
            with np.load(dataset / record["file"], allow_pickle=False) as d:
                for field in ("states", "observations", "masks", "actions", "contexts"):
                    np.testing.assert_array_equal(simulated[field], d[field])
            rows[t["scene_id"]] = simulated
        row = {**rows[t["scene_id"]], "horizon": t["horizon"]}
        pred = {k: np.asarray(v, np.float32) for k, v in t["predictions"].items()}
        fresh = metrics(pred, row, np.ones(4))
        for key, value in fresh.items():
            if isinstance(value, (float, int)):
                assert np.isclose(value, t["metrics"][key], rtol=1e-5, atol=1e-6)
            else:
                assert value == t["metrics"][key]
        key = (i["family"], t["split"], t["horizon"], i["variant"])
        groups.setdefault(key, {}).setdefault(t["scene_id"], []).append(
            fresh["normalized_rollout_mse"]
        )
    comparisons = []
    rng = np.random.default_rng(31011)
    for family in ("motion", "interaction"):
        for panel in ("development_iid", "development_stress"):
            for horizon in (8, 16, 32):
                candidate = groups[family, panel, horizon, "candidate"]
                matched = groups[family, panel, horizon, "matched"]
                assert candidate.keys() == matched.keys()
                a = np.array([np.mean(candidate[k]) for k in sorted(candidate)])
                b = np.array([np.mean(matched[k]) for k in sorted(candidate)])
                idx = rng.integers(len(a), size=(2000, len(a)))
                ratios = 1 - a[idx].mean(1) / np.maximum(b[idx].mean(1), 1e-12)
                comparisons.append(
                    {
                        "family": family,
                        "panel": panel,
                        "horizon": horizon,
                        "scene_clusters": len(a),
                        "relative_reduction": float(1 - a.mean() / b.mean()),
                        "paired_scene_bootstrap95": np.quantile(ratios, [0.025, 0.975]).tolist(),
                    }
                )
    count = sum(sum(len(v) for v in g.values()) for g in groups.values())
    assert count == 780 and plans == 32 and checkpoints == 22
    receipt = {
        "passed": True,
        "forecasts": count,
        "actual_plans": plans,
        "checkpoints": checkpoints,
        "independent_scenes": len(rows),
        "paired_comparisons": comparisons,
        "neural_calls": 0,
        "sealed_evaluation": "NOT RUN",
        "old_final_test_queries": 0,
    }
    atomic_json(root / "verification.json", receipt)
    print(
        json.dumps({k: receipt[k] for k in ("passed", "forecasts", "actual_plans", "checkpoints")})
    )


if __name__ == "__main__":
    import sys

    verify(sys.argv[1])
