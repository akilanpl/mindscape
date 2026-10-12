import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from mindscape.v3.domains import DOMAINS, Task, generate_tasks
from mindscape.v3.experiment import freeze_manifest, read_manifest, run, validate_tasks
from mindscape.v3.kernel import Kernel
from mindscape.v3.memory import Memory
from mindscape.v3.models import TinyMLP
from mindscape.v3.schema import Action, Prediction, State


class AdditionalControls(unittest.TestCase):
    def test_swapped_math_inputs_are_same_identity(self):
        left = Task("mathematics", (2, 3), "train", "unit")
        right = Task("mathematics", (3, 2), "iid", "unit")
        self.assertEqual(left.identity, right.identity)
        with self.assertRaises(ValueError):
            validate_tasks(generate_tasks() + (left, right))

    def test_action_and_task_bounds(self):
        for action in (("submit", True), ("", 1), ("submit", 1.0)):
            with self.assertRaises(ValueError):
                Action(*action)
        for inputs in ((1,), (100, 2), (True, 2)):
            with self.assertRaises(ValueError):
                Task("mathematics", inputs, "train", "unit")
        with self.assertRaises(ValueError):
            Task("unknown", (1, 2), "train", "unit")

    def test_memory_persistence_hash_and_failed_actual_evidence(self):
        kernel = Kernel(TinyMLP(0, True))
        task = Task("mathematics", (2, 3), "train", "unit")
        trace = kernel.run(task, DOMAINS["mathematics"], 0, "unit")
        self.assertFalse(trace["success"])
        self.assertEqual(len(kernel.memory.episodes), 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "memory.json")
            kernel.memory.save(path)
            loaded = Memory.load(path)
            self.assertEqual(loaded.episodes, kernel.memory.episodes)
            data = json.loads(path.read_text())
            data["sha256"] = "tampered"
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                Memory.load(path)
        exp = next(iter(kernel.memory.episodes.values()))
        hypothetical = Prediction(Action("submit", 1), exp.transition.after, 0.5)
        with self.assertRaises(ValueError):
            kernel.memory.add(replace(exp, transition=hypothetical))

    def test_neural_transition_prediction_is_not_execution(self):
        trace = Kernel(TinyMLP(0, True)).run(
            Task("mathematics", (2, 3), "iid", "unit"), DOMAINS["mathematics"], 0, "unit"
        )
        self.assertEqual(trace["events"][3]["value"]["state"]["written"], (0.0,))
        self.assertEqual(trace["events"][4]["value"]["after"]["written"], (-32.0,))
        self.assertEqual(trace["transition_prediction_mse"], 1024)
        self.assertEqual(trace["events"][3]["value"]["kind"], "hypothetical")

    def test_backend_does_not_receive_task_partition_or_private_answers(self):
        model = TinyMLP(1)
        d = DOMAINS["python"]
        task = Task("python", (2, 3, 4), "train", "unit")
        obs = d.observe(task)
        self.assertFalse(hasattr(obs, "split"))
        self.assertFalse(hasattr(obs, "task_id"))
        self.assertFalse(hasattr(obs, "expected"))
        self.assertEqual(
            model.logits(obs, d.candidates(State(obs))),
            model.logits(d.observe(replace(task, split="ood")), d.candidates(State(obs))),
        )

    def test_resume_does_not_call_neural_forward(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp:
            manifest, output = Path(tmp, "manifest.json"), Path(tmp, "run")
            freeze_manifest(manifest)
            first = run(manifest, output, seeds=(0,), conditions=("zero",))
            with patch.object(
                TinyMLP, "logits", side_effect=AssertionError("Unexpected inference")
            ):
                second = run(manifest, output, seeds=(0,), conditions=("zero",))
            self.assertEqual(second["resumed"], first["completed"])

    def test_all_expected_outputs_fit_vocabulary(self):
        for task in generate_tasks():
            d = DOMAINS[task.domain]
            state = State(d.observe(task))
            successes = [
                a for a in d.candidates(state) if d.evaluate(task, d.execute(state, a)).success
            ]
            self.assertEqual(len(successes), 1)
            self.assertEqual(len(d.candidates(state)), 65)

    def test_manifest_missing_split_rejected_even_with_recomputed_hash(self):
        from mindscape.v3.schema import digest

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "manifest.json")
            freeze_manifest(path)
            data = json.loads(path.read_text())
            data["body"]["tasks"] = [t for t in data["body"]["tasks"] if t["split"] != "ood"]
            data["sha256"] = digest(data["body"])
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                read_manifest(path)


if __name__ == "__main__":
    unittest.main()
