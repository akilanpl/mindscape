import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mindscape.v3.physics_world.schema import Action, ObjectSpec, Scene, observe

HAS_ENGINE = importlib.util.find_spec("pymunk") is not None


def blank():
    return Action.from_vector(np.zeros((3, 4)))


@unittest.skipUnless(HAS_ENGINE, "Pinned Pymunk required")
class EnvironmentTests(unittest.TestCase):
    def test_deterministic_trajectories_both_families(self):
        from mindscape.v3.physics_world.data import generate_scene
        from mindscape.v3.physics_world.environment import trajectory

        for family in ("motion", "interaction"):
            scene = generate_scene(family, "train", 0)
            a, b = trajectory(scene), trajectory(scene)
            for key in ("states", "actions", "observations", "masks"):
                np.testing.assert_array_equal(a[key], b[key])

    def test_free_flight_gravity_and_impulse(self):
        from mindscape.v3.physics_world.environment import Environment

        body = ObjectSpec((0, 10), (1, 0), 2, 0.25, 0, 0)
        scene = Scene("motion", "train", 0, 1, "projectile", (body,), 9.8, 0, False, False, 0, 32)
        env = Environment(scene)
        a = np.zeros((3, 4))
        a[0, 2] = 2
        state = env.step(Action.from_vector(a))
        self.assertAlmostEqual(state[0, 2], 2, places=6)
        self.assertAlmostEqual(state[0, 3], -9.8 / 30, places=6)
        self.assertGreater(state[0, 0], 0)

    def test_collision_momentum_and_energy_loss(self):
        from mindscape.v3.physics_world.environment import Environment

        for restitution in (1.0, 0.3):
            bodies = (
                ObjectSpec((-0.5, 2), (1, 0), 1, 0.25, 0, restitution),
                ObjectSpec((0.5, 2), (-1, 0), 2, 0.25, 0, restitution),
            )
            scene = Scene(
                "interaction", "train", 1, 2, "collision", bodies, 0, 0, False, False, 0, 32
            )
            env = Environment(scene)
            initial = env.energy()
            contact = False
            for _ in range(20):
                state = env.step(blank())
                contact |= bool(state[:, 8].sum())
            self.assertTrue(contact)
            self.assertAlmostEqual(float((state[:, 2] * state[:, 4]).sum()), -1, places=5)
            if restitution == 1:
                self.assertAlmostEqual(env.energy(), initial, places=4)
            else:
                self.assertLess(env.energy(), initial)

    def test_spring_coupling(self):
        from mindscape.v3.physics_world.environment import Environment

        bodies = (
            ObjectSpec((-1, 2), (0, 0), 1, 0.25, 0, 1),
            ObjectSpec((1, 2), (0, 0), 1, 0.25, 0, 1),
        )
        scene = Scene("interaction", "train", 1, 2, "spring", bodies, 0, 0, False, False, 5, 32)
        state = Environment(scene).step(blank())
        self.assertGreater(state[0, 2], 0)
        self.assertLess(state[1, 2], 0)

    def test_masking_and_noise_have_no_private_parameters(self):
        from mindscape.v3.physics_world.data import generate_scene
        from mindscape.v3.physics_world.environment import Environment

        scene = generate_scene("motion", "train", 0)
        state = Environment(scene).state()
        public = observe(state, scene, 0)
        self.assertEqual(set(public), {"values", "mask", "context"})
        self.assertTrue((public["values"][:, [4, 6, 7]] == 0).all())
        self.assertTrue((public["mask"][:, [4, 6, 7]] == 0).all())
        np.testing.assert_array_equal(public["values"], observe(state, scene, 0)["values"])

    def test_freeze_integrity_and_nested_budgets(self):
        from mindscape.v3.physics_world.data import freeze, load, training_arrays

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            m = freeze(root, {"train": 3, "validation": 1, "iid": 1, "ood": 2})
            self.assertEqual(len({r["identity"] for r in m["records"]}), 14)
            small = training_arrays(root, "motion", 1)
            large = training_arrays(root, "motion", 3)
            self.assertEqual(small["trajectory_ids"], large["trajectory_ids"][:1])
            self.assertEqual(load(root)["hash"], m["hash"])
            first = root / m["records"][0]["file"]
            first.write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                load(root)

    def test_action_bounds(self):
        with self.assertRaises(ValueError):
            Action.from_vector(np.full((3, 4), 21))
        with self.assertRaises(ValueError):
            Action.from_vector(np.zeros((4, 4)))


class NonNeuralBoundaryTests(unittest.TestCase):
    def test_predictor_has_no_simulator_dependency(self):
        import ast

        for name in ("model.py", "planning.py", "evaluation.py"):
            tree = ast.parse(
                (Path(__file__).parents[1] / "src/mindscape/v3/physics_world" / name).read_text()
            )
            modules = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
            self.assertFalse(any("environment" in m or "pymunk" in m for m in modules))

    def test_observation_cannot_contain_future_state(self):
        from mindscape.v3.physics_world.evaluation import rollout

        with self.assertRaisesRegex(ValueError, "Private"):
            rollout(None, {"states": np.zeros(1)}, "persistence")

    def test_constant_velocity_and_persistence(self):
        from mindscape.v3.physics_world.baselines import predict

        h = np.zeros((1, 4, 3, 10))
        h[:, :, :, 9] = 1
        h[:, :, :, 2] = 2
        m = np.ones_like(h)
        p, _ = predict(h, m, np.zeros((1, 3, 4)), np.zeros((1, 4)), "constant_velocity")
        self.assertAlmostEqual(p[0, 0, 0], 2 / 30)
        p, _ = predict(h, m, np.zeros((1, 3, 4)), np.zeros((1, 4)), "persistence")
        self.assertEqual(p[0, 0, 0], 0)
