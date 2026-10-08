from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import random
import unittest

from mindscape.core.config import load_config
from mindscape.core.schema import Action, EvidenceKind, Observation
from mindscape.environments.multiplication.environment import MultiplicationEnvironment
from mindscape.verification.verifier import verify


class FoundationTests(unittest.TestCase):
    def solve(self, a, b):
        env = MultiplicationEnvironment()
        env.reset((a, b))
        trajectory = env.run()
        self.assertEqual(env.get_state().answer, a * b)
        self.assertTrue(env.is_goal_reached())
        self.assertTrue(verify(trajectory, a * b).verified)
        self.assertEqual(env.valid_actions(), ())
        return env, trajectory

    def test_initial_state(self):
        env = MultiplicationEnvironment()
        s = env.reset((19, 3))
        self.assertEqual(s.left_digits, (9, 1))
        self.assertEqual(s.right_digits, (3,))
        self.assertEqual((s.position, s.carry, s.row_value), (0, 0, 0))
        self.assertEqual(len(s.entities), 2)
        self.assertEqual(s.relations[0].predicate, "multiplied_by")
        self.assertEqual(s.goal, env.get_goal())
        self.assertFalse(env.is_goal_reached())

    def test_requires_reset(self):
        env = MultiplicationEnvironment()
        for method in [env.get_state, env.get_trajectory, env.valid_actions, env.get_goal]:
            with self.assertRaises(RuntimeError):
                method()

    def test_input_validation(self):
        for operands in [(True, 3), (1.0, 2), ("1", 3), (1,), [1, 2]]:
            with self.assertRaises(ValueError):
                Observation(operands)

    def test_evidence_categories(self):
        self.assertEqual(len(set(EvidenceKind)), 5)
        with self.assertRaises(ValueError):
            Observation((1, 2), kind=EvidenceKind.PREDICTION)

    def test_frozen_state(self):
        env = MultiplicationEnvironment()
        state = env.reset((19, 3))
        with self.assertRaises(FrozenInstanceError):
            state.carry = 10

    def test_carry_generation_and_propagation(self):
        env = MultiplicationEnvironment()
        env.reset((19, 3))
        first = env.step(env.valid_actions()[0])
        self.assertEqual((first.result.value, first.result.written_digit,
                          first.state_after.carry), (27, 7, 2))
        second = env.step(env.valid_actions()[0])
        self.assertEqual((second.result.value, second.state_after.row_value,
                          second.state_after.carry), (5, 57, 0))
        self.assertEqual(first.state_after, second.state_before)
        env.run()
        self.assertTrue(verify(env.get_trajectory(), 57).verified)

    def test_remaining_carry(self):
        env, trajectory = self.solve(99, 9)
        flush = next(t for t in trajectory.transitions if t.action.name == "flush")
        self.assertEqual(flush.state_before.carry, 8)
        self.assertEqual(flush.result.value, 891)

    def test_partial_rows(self):
        _, trajectory = self.solve(123, 45)
        rows = [t.result.value for t in trajectory.transitions
                if t.action.name == "accumulate"]
        self.assertEqual(rows, [615, 5535])

    def test_signed_and_zero(self):
        for pair in [(0, 123), (123, 0), (-19, 3), (19, -3), (-19, -3), (0, -9)]:
            with self.subTest(pair=pair):
                self.solve(*pair)

    def test_exhaustive_small_integer_domain(self):
        for a in range(-20, 21):
            for b in range(-20, 21):
                self.solve(a, b)

    def test_seeded_large_oracle_cases(self):
        rng = random.Random(42)
        for _ in range(200):
            self.solve(rng.randint(-99999999, 99999999), rng.randint(-99999999, 99999999))

    def test_invalid_action_preserves_state(self):
        env = MultiplicationEnvironment()
        before = env.reset((19, 3))
        with self.assertRaises(ValueError):
            env.step(Action("finish", 0, 0))
        self.assertEqual(env.get_state(), before)
        self.assertEqual(len(env.get_trajectory().transitions), 0)

    def test_stale_and_terminal_actions(self):
        env = MultiplicationEnvironment()
        env.reset((19, 3))
        action = env.valid_actions()[0]
        env.step(action)
        with self.assertRaises(ValueError):
            env.step(action)
        env.run()
        with self.assertRaises(ValueError):
            env.step(Action("finish", 0, 1))

    def test_reset_clears_episode(self):
        env, _ = self.solve(19, 3)
        env.reset((2, 2))
        self.assertEqual(env.get_trajectory().transitions, ())
        self.assertFalse(env.is_goal_reached())

    def test_determinism(self):
        _, a = self.solve(908, 79)
        _, b = self.solve(908, 79)
        self.assertEqual(a, b)

    def test_initial_state_tamper(self):
        _, trajectory = self.solve(19, 3)
        corrupt = replace(trajectory, initial_state=replace(trajectory.initial_state, carry=1))
        self.assertFalse(verify(corrupt).trajectory_valid)

    def test_every_transition_field_checked(self):
        _, trajectory = self.solve(19, 3)
        t = trajectory.transitions[0]
        mutations = [
            replace(t, state_before=replace(t.state_before, carry=1)),
            replace(t, action=Action("finish", 0, 0)),
            replace(t, event=replace(t.event, detail="unsupported")),
            replace(t, result=replace(t.result, value=999)),
            replace(t, state_after=replace(t.state_after, row_value=99)),
            replace(t, result=replace(t.result, kind=EvidenceKind.HYPOTHETICAL_RESULT)),
        ]
        for mutation in mutations:
            corrupted = replace(trajectory, transitions=(mutation,) + trajectory.transitions[1:])
            self.assertFalse(verify(corrupted).verified)

    def test_missing_reordered_duplicate_steps(self):
        _, trajectory = self.solve(19, 3)
        ts = trajectory.transitions
        for mutated in [ts[1:], (ts[1], ts[0]) + ts[2:], ts + (ts[-1],)]:
            self.assertFalse(verify(replace(trajectory, transitions=mutated)).verified)

    def test_incomplete_prefix(self):
        _, trajectory = self.solve(19, 3)
        outcome = verify(replace(trajectory, transitions=trajectory.transitions[:-1]), 57)
        self.assertTrue(outcome.trajectory_valid)
        self.assertFalse(outcome.final_answer_valid)
        self.assertFalse(outcome.goal_reached)

    def test_wrong_claimed_answer(self):
        _, trajectory = self.solve(19, 3)
        outcome = verify(trajectory, 58)
        self.assertTrue(outcome.trajectory_valid)
        self.assertTrue(outcome.goal_reached)
        self.assertFalse(outcome.verified)

    def test_configuration(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(load_config(root / "configs/base.toml")["seed"], 42)


if __name__ == "__main__":
    unittest.main()
