import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from mindscape.v3.physics_world.schema import digest
from mindscape.v3.physics_world.storage import RunLock, atomic_json

HAS_ENGINE = importlib.util.find_spec("pymunk") is not None


class ProvenanceTests(unittest.TestCase):
    def test_atomic_identity_and_writer_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            lock = RunLock(root, {"seed": 11})
            try:
                with self.assertRaises(BlockingIOError):
                    RunLock(root, {"seed": 11})
            finally:
                lock.close()
            with self.assertRaises(ValueError):
                RunLock(root, {"seed": 23})

    def test_json_nonfinite_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "row.json"
            with self.assertRaises(ValueError):
                atomic_json(path, {"x": float("nan")})
            self.assertFalse(path.exists())

    def test_mps_model_does_not_construct_on_cpu(self):
        from mindscape.v3.physics_world.model import WorldModel

        with (
            patch(
                "mindscape.v3.physics_world.model.require_mps", side_effect=RuntimeError("blocked")
            ),
            patch("torch.nn.Linear") as linear,
        ):
            with self.assertRaises(RuntimeError):
                WorldModel()
            linear.assert_not_called()

    def test_public_v3_record_contains_missing_fields(self):
        from mindscape.v3.physics_world.adapter import represent
        from mindscape.v3.schema import decode, encode

        v = np.zeros((3, 10))
        v[0, 9] = 1
        m = np.zeros_like(v)
        m[0, [0, 1, 5, 9]] = 1
        record = represent({"values": v, "mask": m, "context": np.zeros(4)})
        self.assertTrue(record.partial)
        self.assertIn("object0.mass", record.missing)
        self.assertEqual(decode(encode(record)), record)
        with self.assertRaises(ValueError):
            represent({"values": v, "mask": m, "context": np.zeros(4), "private_mass": 2})

    def test_analysis_requires_pairing(self):
        from mindscape.v3.physics_world.analysis import analyze

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            atomic_json(root / "identity.json", {"primary_required": 6000})
            trace = {
                "scene_id": "a",
                "split": "ood",
                "metrics": {"normalized_rollout_mse": 1.0, "one_step_mse": 1.0, "finite": True},
            }
            for name, scene in [("mindscape", "a"), ("matched", "b")]:
                t = {**trace, "scene_id": scene}
                folder = root / name
                folder.mkdir()
                atomic_json(
                    folder / (digest(name) + ".json"),
                    {
                        "identity": {"family": "motion", "seed": 11, "budget": 10, "variant": name},
                        "trace_hash": digest(t),
                        "trace": t,
                    },
                )
            with self.assertRaises(ValueError):
                analyze(root, root / "analysis.json")

    def test_analysis_does_not_count_frames_as_scenes(self):
        from mindscape.v3.physics_world.analysis import analyze

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            atomic_json(root / "identity.json", {"primary_required": 6000})
            trace = {
                "scene_id": "a",
                "split": "ood",
                "horizon": 64,
                "metrics": {"normalized_rollout_mse": 2.0, "one_step_mse": 1.0, "finite": True},
            }
            atomic_json(
                root / (digest("a") + ".json"),
                {
                    "identity": {
                        "family": "motion",
                        "seed": 11,
                        "budget": 10,
                        "variant": "mindscape",
                    },
                    "trace_hash": digest(trace),
                    "trace": trace,
                },
            )
            result = analyze(root, root / "analysis.json")
            self.assertEqual(result["primary_completed"], 1)
            self.assertFalse(result["primary_complete"])


@unittest.skipUnless(HAS_ENGINE, "Pinned engine required")
class PhysicsControlTests(unittest.TestCase):
    def test_all_seven_ood_strata_and_identifiers(self):
        from mindscape.v3.physics_world.data import OOD, generate_scene

        for family in ("motion", "interaction"):
            scenes = [generate_scene(family, "ood", i) for i in range(7)]
            self.assertEqual({s.ood_condition for s in scenes}, set(OOD))
            self.assertEqual(len({s.identity for s in scenes}), 7)
            for s in scenes:
                self.assertNotEqual(s.identity, generate_scene(family, "iid", s.index).identity)

    def test_friction_reduces_tangential_motion(self):
        from mindscape.v3.physics_world.environment import Environment
        from mindscape.v3.physics_world.schema import Action, ObjectSpec, Scene

        end = []
        for friction in (0, 1):
            body = ObjectSpec((0, 0.27), (2, 0), 1, 0.25, friction, 0)
            env = Environment(
                Scene("motion", "train", 1, 1, "friction", (body,), 9.8, 0, False, False, 0, 32)
            )
            for _ in range(30):
                state = env.step(Action.from_vector(np.zeros((3, 4))))
            end.append(state[0, 2])
        self.assertLess(end[1], end[0])

    def test_approved_policy_labels_use_only_training(self):
        from mindscape.v3.physics_world.data import freeze
        from mindscape.v3.physics_world.policy import teacher_data

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = freeze(root / "data", {"train": 1, "validation": 1, "iid": 1, "ood": 1})
            rows = teacher_data(manifest, "motion", root / "teachers")
            training = {
                r["identity"]
                for r in manifest["records"]
                if r["scene"]["family"] == "motion" and r["scene"]["split"] == "train"
            }
            self.assertEqual(set(rows["training_ids"]), training)
            self.assertEqual(len(rows["rows"]), 1)

    def test_zero_and_random_wrong_predictions_do_not_become_ground_truth(self):
        from mindscape.v3.physics_world.data import generate_scene
        from mindscape.v3.physics_world.environment import trajectory
        from mindscape.v3.physics_world.evaluation import metrics

        scene = generate_scene("motion", "iid", 1)
        row = trajectory(scene)
        row["horizon"] = 32
        zero = {
            "states": np.zeros((32, 3, 4)),
            "variance": np.ones((32, 3, 4)),
            "contacts": np.zeros((32, 3)),
        }
        self.assertGreater(metrics(zero, row, np.ones(4))["normalized_rollout_mse"], 0.1)

    def test_pilot_seed_is_disjoint_from_main(self):
        from mindscape.v3.physics_world.data import generate_scene

        main = {
            generate_scene(family, split, i, 20261012).identity
            for family in ("motion", "interaction")
            for split in ("train", "validation", "iid", "ood")
            for i in range(3)
        }
        pilot = {
            generate_scene(family, split, i, 20261013).identity
            for family in ("motion", "interaction")
            for split in ("train", "validation", "iid", "ood")
            for i in range(3)
        }
        self.assertFalse(main.intersection(pilot))

    def test_checkpoint_metadata_is_safe_primitive(self):
        from types import SimpleNamespace

        import torch

        from mindscape.v3.physics_world.training import checkpoint

        model = SimpleNamespace(
            state_dict=lambda: {"fixture": torch.tensor(1.0)},
            memory_keys=None,
            memory_residuals=None,
            memory_active=None,
        )
        optimizer = SimpleNamespace(state_dict=lambda: {"state": {}, "param_groups": []})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint-index.json"
            checkpoint(path, model, optimizer, 1, {"torch": torch.__version__})
            index = json.loads(path.read_text())
            saved = torch.load(Path(tmp) / index["file"], weights_only=True, map_location="cpu")
            self.assertIs(type(saved["identity"]["torch"]), str)

    def test_genuine_native_checkpoint_continuation_when_available(self):
        import importlib.util

        root = Path("results/v3_physics/native-pilot-1791811351750339000")
        if not root.exists():
            self.skipTest("Native MPS evidence unavailable; CI cannot certify GPU execution")
        spec = importlib.util.spec_from_file_location(
            "checkpoint_regression", "scripts/v3_physics/checkpoint_regression.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            receipt = module.inspect(root, Path(tmp) / "receipt.json")
        self.assertTrue(receipt["passed"])
        self.assertEqual(receipt["checkpoints_loaded"], 20)
        self.assertFalse(receipt["neural_cpu_execution"])
