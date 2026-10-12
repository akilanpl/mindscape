"""Direct replication against frozen V1 source; no checkpoint modification."""

import ast
import unittest
from pathlib import Path


class HistoricalConfound(unittest.TestCase):
    def test_singleton_mask_is_an_oracle_without_weights(self):
        from mindscape.core.schema import Action
        from mindscape.environments.multiplication.environment import MultiplicationEnvironment

        # Read the frozen literal vocabulary without importing its optional NumPy dependency.
        path = Path(__file__).resolve().parents[2] / "src/mindscape/models/encoding.py"
        tree = ast.parse(path.read_text())
        action_names = next(
            ast.literal_eval(n.value)
            for n in tree.body
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "ACTION_NAMES" for t in n.targets)
        )

        for operands in ((12, 34), (-1234, 321), (7, 4321)):
            masked = MultiplicationEnvironment()
            masked.reset(operands)
            counts = []
            while not masked.is_goal_reached():
                candidates = masked.valid_actions()
                counts.append(len(candidates))
                # Argmax of all-zero logits after exact masking always chooses this one action.
                masked.step(candidates[0])
            self.assertTrue(masked.is_goal_reached())
            self.assertEqual(set(counts), {1})
            unmasked = MultiplicationEnvironment()
            unmasked.reset(operands)
            for _ in range(256):
                if unmasked.is_goal_reached():
                    break
                state = unmasked.get_state()
                # Same deterministic all-zero tie break over the full action vocabulary.
                action = Action(action_names[0], state.position, state.row_position)
                if action not in unmasked.valid_actions():
                    break
                unmasked.step(action)
            self.assertFalse(unmasked.is_goal_reached())


if __name__ == "__main__":
    unittest.main()
