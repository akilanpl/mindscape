import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from mindscape.v3.physics_world.schema import digest
from mindscape.v3_1.physics.data import create, windows


class Boundaries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name) / "data"
        cls.protocol = json.loads(Path("configs/v3_1_physics/pilot.json").read_text())
        cls.manifest = create(cls.root, cls.protocol)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_physical_task_identity_disjoint(self):
        physical = []
        for r in self.manifest["records"]:
            scene = {
                k: v
                for k, v in r["scene"].items()
                if k not in ("seed", "split", "index", "ood_condition")
            }
            physical.append(digest(scene))
        self.assertEqual(len(physical), len(set(physical)))
        self.assertEqual(len(physical), 128)

    def test_training_multiobject_exposure_and_windows(self):
        approved = [
            r
            for r in self.manifest["records"]
            if r["scene"]["family"] == "motion" and r["scene"]["split"] == "train"
        ]
        self.assertEqual({len(r["scene"]["objects"]) for r in approved}, {1, 2})
        batch = windows(self.root, "motion", 8)
        self.assertEqual(set(batch["ids"]), {r["identity"] for r in approved})
        self.assertEqual(batch["targets"].shape[1:], (8, 3, 4))
        self.assertEqual(batch["history"].shape[1:], (4, 3, 10))

    def test_sealed_trajectories_never_opened_by_training(self):
        approved = {r["file"] for r in self.manifest["records"] if r["scene"]["split"] == "train"}
        original = np.load

        def guard(path, *args, **kwargs):
            self.assertIn(Path(path).name, approved)
            return original(path, *args, **kwargs)

        with patch("numpy.load", side_effect=guard):
            windows(self.root, "interaction", 8)

    def test_fail_before_cpu_neural_construction(self):
        from mindscape.v3_1.physics.model import Dynamics

        with (
            patch("mindscape.v3_1.physics.model.require_mps", side_effect=RuntimeError("MPS")),
            patch("torch.nn.Linear", side_effect=AssertionError("CPU construction")),
            self.assertRaisesRegex(RuntimeError, "MPS"),
        ):
            Dynamics("candidate", 11)

    def test_no_engine_in_predictive_modules(self):
        for filename in ("model.py", "evaluation.py"):
            tree = ast.parse((Path("src/mindscape/v3_1/physics") / filename).read_text())
            imports = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
            self.assertFalse(any("environment" in x or "pymunk" in x for x in imports))

    def test_feature_dimension_matches_declared_scaling(self):
        from mindscape.v3_1.physics.model import FEATURES

        self.assertEqual(FEATURES, 2 * 10 + 2 * 10 + 4 + 4 + 6)

    def test_private_prediction_fields_rejected(self):
        from mindscape.v3_1.physics.evaluation import rollout

        with patch("mindscape.v3_1.physics.evaluation.require_mps"), self.assertRaises(ValueError):
            rollout(None, {"private_states": np.zeros(4)})

    def test_manifest_change_rejected(self):
        from mindscape.v3.physics_world.data import load

        path = self.root / "manifest.json"
        original = path.read_text()
        altered = json.loads(original)
        altered["history"] = 7
        path.write_text(json.dumps(altered))
        try:
            with self.assertRaises(ValueError):
                load(self.root)
        finally:
            path.write_text(original)

    def test_derived_memory_checkpoint_creates_parent_and_safe_roundtrip(self):
        from types import SimpleNamespace

        import torch

        from mindscape.v3_1.physics.checkpoints import save_derived_memory

        # Serialization tensors only; no CPU neural model or optimizer computation.
        model = SimpleNamespace(
            state_dict=lambda: {"weights": torch.tensor([1.0, 2.0])},
            memory_keys=torch.tensor([[1.0, 3.0]]),
            memory_residuals=torch.tensor([[0.1, 0.2]]),
            memory_active={"split": "train", "trajectory_ids": ["approved"]},
        )
        optimizer = SimpleNamespace(state_dict=lambda: {"state": {}, "param_groups": []})
        identity = {"variant": "memory", "derived_from": "parent-checkpoint-hash"}
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "new/family/seed/memory"
            self.assertFalse(directory.exists())
            pointer = save_derived_memory(directory, model, optimizer, 160, identity)
            index = json.loads(pointer.read_text())
            saved = torch.load(directory / index["file"], map_location="cpu", weights_only=True)
            self.assertEqual(saved["step"], 160)
            self.assertEqual(saved["identity"], identity)
            self.assertEqual(saved["memory_active"], model.memory_active)
            self.assertTrue(torch.equal(saved["state"]["weights"], model.state_dict()["weights"]))
            self.assertTrue(torch.equal(saved["memory_keys"], model.memory_keys))
            self.assertTrue(torch.equal(saved["memory_residuals"], model.memory_residuals))
