import ast
import json
import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from mindscape.v3.math_learning.analysis import mcnemar_exact
from mindscape.v3.math_learning.data import freeze, load
from mindscape.v3.math_learning.environment import (
    Claim,
    LocalState,
    Problem,
    demonstration,
    execute_local,
    expected,
    run_problem,
    teacher_local,
)
from mindscape.v3.math_learning.experiment import Oracle, evaluate
from mindscape.v3.math_learning.learner import Predictor, TransitionNet, local_features
from mindscape.v3.math_learning.training import build_training, train
from mindscape.v3.math_learning.verify import verify


class EnvironmentTests(unittest.TestCase):
    def test_teacher_and_multistep_scaffold_agree_with_independent_arithmetic(self):
        rng = random.Random(11)
        for _ in range(100):
            a, b, c = (rng.randint(-999, 999) for _ in range(3))
            for operations in (
                ("add",),
                ("sub",),
                ("mul",),
                ("add", "mul"),
                ("sub", "add"),
                ("mul", "sub"),
            ):
                operands = (a, b, c) if len(operations) == 2 else (a, b)
                p = Problem(operands, operations, "unit", "test")
                trace = run_problem(p, Oracle())
                self.assertTrue(trace["strict_rollout"], (p, trace))
                self.assertEqual(int(trace["answer"]), expected(p))

    def test_exhaustive_local_vocab_teacher(self):
        for op in ("add", "sub", "mul"):
            for a in range(10):
                for b in range(10):
                    for carry in range(10):
                        claim = teacher_local(LocalState(op, a, b, carry))
                        self.assertTrue(0 <= claim.digit <= 9)
                        self.assertEqual(
                            execute_local(LocalState(op, a, b, carry), claim)["incoming"],
                            claim.carry,
                        )

    def test_wrong_claim_writes_without_teacher(self):
        state = LocalState("add", 9, 9, 1)
        wrong = Claim(2, 0, 3, 7, 4, 8)
        with patch(
            "mindscape.v3.math_learning.environment.teacher_local",
            side_effect=AssertionError("oracle leak"),
        ):
            result = execute_local(state, wrong)
        self.assertEqual(result["written_digit"], 4)
        self.assertEqual(result["incoming"], 8)

    def test_verifier_only_after_rollout(self):
        p = Problem((12, 34), ("add",), "unit", "test")

        class Spy:
            count = 0

            def predict_local(self, state):
                self.count += 1
                return Claim(0, 0, 0, 0, 0, 0), {"consistency": 0.0}

            def predict_sign(self, visible):
                return 0, {}

        spy = Spy()
        original = teacher_local

        def judge(state):
            self.assertEqual(spy.count, 3)
            return original(state)

        with patch("mindscape.v3.math_learning.environment.teacher_local", side_effect=judge):
            trace = run_problem(p, spy)
        self.assertFalse(trace["success"])

    def test_budget_and_input_guards(self):
        with self.assertRaises(RuntimeError):
            run_problem(Problem((99, 99), ("mul",), "unit", "t"), Oracle(), 1)
        with self.assertRaises(ValueError):
            Problem((1, 2), ("divide",), "unit", "t")
        with self.assertRaises(ValueError):
            LocalState("add", 10, 0, 0)
        with self.assertRaises(ValueError):
            demonstration(Problem((1, 2), ("add",), "ood", "t"))


class LeakageAndLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)

    def test_frozen_splits_and_encrypted_lockbox(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp, "dataset")
            freeze(root, 123)
            manifest, tasks = load(root)
            self.assertEqual(len(tasks["train"]), 1000)
            self.assertNotIn("lockbox", tasks)
            self.assertNotIn(b"operands", (root / "lockbox.aesgcm").read_bytes())
            self.assertEqual(len(manifest["body"]["lockbox_ids"]), 90)
            with self.assertRaises(FileExistsError):
                freeze(root, 123)
            data = json.loads((root / "manifest.json").read_text())
            data["sha256"] = "tamper"
            (root / "manifest.json").write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                load(root)

    def test_task_identity_commutation_and_train_gate(self):
        a = Problem((2, 3), ("add",), "train", "t")
        b = replace(a, operands=(3, 2), split="ood")
        self.assertEqual(a.identity, b.identity)
        with self.assertRaises(ValueError):
            build_training([b])
        self.assertNotEqual(
            replace(a, operations=("sub",)).identity, replace(b, operations=("sub",)).identity
        )

    def test_neural_module_has_no_arithmetic_teacher_reference(self):
        import mindscape.v3.math_learning.learner as module

        tree = ast.parse(Path(module.__file__).read_text())
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        self.assertNotIn("teacher_local", names)
        self.assertNotIn("expected", names)

    def test_actual_gradients_and_exact_optimizer_resume(self):
        problems = [Problem((a, b), ("add",), "train", "t") for a, b in ((2, 3), (4, 5), (6, 7))]
        arrays, memory, ledger, _ = build_training(problems)
        with tempfile.TemporaryDirectory() as tmp:
            whole, _ = train(arrays, ledger, Path(tmp, "whole"), 0, 8)
            train(arrays, ledger, Path(tmp, "resumed"), 0, 8, stop_after=3)
            resumed, receipt = train(arrays, ledger, Path(tmp, "resumed"), 0, 8)
            for key, value in whole.state_dict().items():
                self.assertTrue(torch.equal(value, resumed.state_dict()[key]))
            initial = TransitionNet(0)
            self.assertTrue(
                any(
                    not torch.equal(v, initial.state_dict()[k])
                    for k, v in whole.state_dict().items()
                )
            )
            _, again = train(arrays, ledger, Path(tmp, "resumed"), 0, 8)
            self.assertEqual(receipt["checkpoint_sha256"], again["checkpoint_sha256"])
            self.assertEqual(ledger["distinct_examples"], 3)
            self.assertEqual(
                receipt["active_intermediate_labels"], ledger["intermediate_scalar_labels"]
            )
            self.assertGreater(len(memory), 0)

    def test_equalinfo_ablations_share_weights_and_no_eval_memory_write(self):
        model = TransitionNet(3)
        memory = {("add", 2, 3, 0): teacher_local(LocalState("add", 2, 3, 0))}
        before = dict(memory)
        predictors = [
            Predictor(model, memory, True),
            Predictor(model, {}, True),
            Predictor(model, memory, False),
        ]
        for predictor in predictors:
            p = Problem((12, 34), ("add",), "unit", "t")
            run_problem(p, predictor)
            self.assertIs(predictor.model, model)
        self.assertEqual(memory, before)

    def test_zero_and_random_are_not_rescued_by_controller(self):
        problems = [
            Problem((12, 34), ("add",), "unit", "t"),
            Problem((99, 1), ("sub",), "unit", "t"),
            Problem((12, 13), ("mul",), "unit", "t"),
        ]
        for zero in (False, True):
            model = TransitionNet(0)
            if zero:
                for p in model.parameters():
                    p.data.zero_()
            predictor = Predictor(model, {}, False)
            traces = [run_problem(p, predictor) for p in problems]
            self.assertFalse(all(t["success"] for t in traces))
            for trace in traces:
                for event in trace["events"]:
                    if "diagnostics" in event:
                        self.assertGreater(event["diagnostics"]["syntactic_candidate_count"], 1)
                        self.assertFalse(event["diagnostics"]["semantic_mask"])

    def test_memory_only_retrieves_demonstrated_contexts(self):
        model = TransitionNet(0)
        memory = {("add", 2, 3, 0): teacher_local(LocalState("add", 2, 3, 0))}
        predictor = Predictor(model, memory, True)
        _, yes = predictor.predict_local(LocalState("add", 2, 3, 0))
        _, no = predictor.predict_local(LocalState("add", 8, 9, 0))
        self.assertTrue(yes["retrieved_training_instance"])
        self.assertFalse(no["retrieved_training_instance"])
        self.assertLessEqual(len(yes["candidate_claims"]), 4)
        self.assertEqual(local_features(LocalState("add", 2, 3, 0)).shape, (320,))

    def test_evaluation_resume_integrity_and_replay(self):
        p = Problem((12, 34), ("add",), "unit", "t")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp, "eval-procedural-n10-s0")
            predictor = Predictor(TransitionNet(0), {}, False)
            rows = evaluate(
                [p], predictor, root, "procedural", {"condition": "procedural", "seed": 0}
            )
            with patch.object(
                predictor, "predict_local", side_effect=AssertionError("extra forward")
            ):
                self.assertEqual(
                    rows,
                    evaluate(
                        [p], predictor, root, "procedural", {"condition": "procedural", "seed": 0}
                    ),
                )
            self.assertEqual(verify(tmp)["executions"], 1)
            (root / "contexts.json.gz").write_bytes(b"bad")
            with self.assertRaises(ValueError):
                evaluate([p], predictor, root, "procedural", {"condition": "procedural", "seed": 0})

    def test_exact_paired_statistics(self):
        self.assertEqual(mcnemar_exact(0, 0), 1)
        self.assertAlmostEqual(mcnemar_exact(5, 0), 0.0625)
        self.assertEqual(mcnemar_exact(5, 5), 1)


class AdditionalMathSecurity(unittest.TestCase):
    def test_pretrained_residual_adapter_receives_gradients(self):
        class FakeEncoder:
            size = 4

            def __init__(self):
                self.hashes = {"synthetic": "unit-only"}

            tokens = 0
            parameter_count = 0

            def encode(self, x):
                return torch.from_numpy(np.asarray(x, dtype=np.float32)[:, :4])

        problems = [Problem((2, 3), ("add",), "train", "t")]
        arrays, _, ledger, _ = build_training(problems)
        with tempfile.TemporaryDirectory() as tmp:
            initial = TransitionNet(2, 4)
            model, result = train(arrays, ledger, tmp, 2, 4, encoder=FakeEncoder())
            self.assertGreater(result["adapter_parameters"], 0)
            self.assertFalse(torch.equal(initial.adapter_down.weight, model.adapter_down.weight))
            self.assertFalse(torch.equal(initial.adapter_up.weight, model.adapter_up.weight))

    def test_checkpoint_tamper_rejected(self):
        arrays, _, ledger, _ = build_training([Problem((2, 3), ("add",), "train", "t")])
        with tempfile.TemporaryDirectory() as tmp:
            train(arrays, ledger, tmp, 0, 2)
            path = Path(tmp, "checkpoint.pt")
            path.write_bytes(path.read_bytes() + b"tampered")
            with self.assertRaises(ValueError):
                train(arrays, ledger, tmp, 0, 2)

    def test_common_v3_prediction_interface_stays_hypothetical(self):
        from mindscape.v3.schema import Action, Goal, Observation, State

        predictor = Predictor(TransitionNet(0), {}, False)
        observation = Observation(
            "mathematics", (0.0, 2.0, 3.0, 0.0), Goal("propose local transition")
        )
        actions = tuple(Action("write", i) for i in range(10))
        scores = predictor.logits(observation, actions)
        self.assertEqual(len(scores), 10)
        prediction = predictor.predict_transition(State(observation), actions[0], 0.1)
        self.assertEqual(prediction.kind, "hypothetical")
        self.assertEqual(len(prediction.state.written), 2)


if __name__ == "__main__":
    unittest.main()
