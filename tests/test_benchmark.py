from dataclasses import asdict, replace
import copy
import json
from pathlib import Path
import tempfile
import unittest

from mindscape.data.backends import get_backend, stable_hash
from mindscape.data.generation import generate, load_dataset, save_dataset
from mindscape.data.splits import nested_subsets, validate_dataset
from mindscape.evaluation.evaluator import evaluate
from mindscape.evaluation.metrics import aggregate, data_efficiency_ratio, n_star
from mindscape.evaluation.runner import run
from mindscape.models.base import Prediction
from mindscape.models.reference import ReferenceModel


def config(seed=42):
    return {"environment": "integer_multiplication", "seed": seed, "budgets": [2, 5, 10],
            "splits": {
                "train": {"count": 12, "structures": ["1x1", "2x1"]},
                "validation": {"count": 5, "structures": ["1x1", "2x1"]},
                "test": {"count": 6, "structures": ["1x1", "2x1"]},
                "ood_test": {"count": 6, "structures": ["3x2", "3x3"]}}}


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = config()
        cls.splits = generate(cls.config)
        cls.backend = get_backend(cls.config["environment"])

    def test_identical_seed(self):
        self.assertEqual(self.splits, generate(config()))

    def test_different_seeds(self):
        self.assertNotEqual(self.splits, generate(config(43)))

    def test_targets_verified(self):
        for rows in self.splits.values():
            for example in rows:
                self.backend.validate(example)
                a, b = example.observation["operands"]
                self.assertEqual(example.target_answer, a * b)
                self.assertEqual(example.metadata["difficulty"]["step_count"], len(example.target_trajectory))

    def test_digit_range_configuration(self):
        cfg = config()
        cfg["splits"]["train"] = {"count": 12, "min_digits_a": 1, "max_digits_a": 2,
                                  "min_digits_b": 1, "max_digits_b": 1}
        self.assertTrue(validate_dataset(generate(cfg), cfg))

    def test_invalid_digit_ranges(self):
        with self.assertRaises(ValueError):
            self.backend.structures({"min_digits_a": 3, "max_digits_a": 1,
                                     "min_digits_b": 1, "max_digits_b": 1})

    def test_disjoint_identity(self):
        seen = set()
        for rows in self.splits.values():
            for example in rows:
                identity = self.backend.identity(example.observation)
                self.assertNotIn(identity, seen)
                seen.add(identity)

    def test_structural_isolation(self):
        train = {self.backend.structure_identity(e.metadata["structural_category"]) for e in self.splits["train"]}
        ood = {self.backend.structure_identity(e.metadata["structural_category"]) for e in self.splits["ood_test"]}
        self.assertFalse(train & ood)

    def test_swapped_ood_rejected(self):
        cfg = config()
        cfg["splits"]["ood_test"]["structures"] = ["1x2"]
        with self.assertRaises(ValueError):
            generate(cfg)

    def test_duplicate_rejected(self):
        splits = copy.deepcopy(self.splits)
        splits["train"][1] = splits["train"][0]
        with self.assertRaises(ValueError):
            validate_dataset(splits, self.config)

    def test_swapped_operand_leakage(self):
        splits = copy.deepcopy(self.splits)
        e = splits["train"][0]
        observation = {**e.observation, "operands": list(reversed(e.observation["operands"]))}
        splits["test"][0] = self.backend.example(observation, 42, "test")
        with self.assertRaises(ValueError):
            validate_dataset(splits, self.config)

    def test_wrong_membership(self):
        splits = copy.deepcopy(self.splits)
        splits["test"][0].metadata["split"] = "train"
        with self.assertRaises(ValueError):
            validate_dataset(splits, self.config)

    def test_trace_leakage_or_corruption(self):
        splits = copy.deepcopy(self.splits)
        splits["test"][0].target_trajectory[:] = splits["train"][0].target_trajectory
        with self.assertRaises(ValueError):
            validate_dataset(splits, self.config)

    def test_bad_target(self):
        splits = copy.deepcopy(self.splits)
        splits["test"][0] = replace(splits["test"][0], target_answer=-999)
        with self.assertRaises(ValueError):
            validate_dataset(splits, self.config)

    def test_nested_subsets(self):
        a = nested_subsets(self.splits["train"], [2, 5, 10], 42)
        b = nested_subsets(self.splits["train"], [2, 5, 10], 42)
        self.assertEqual(a, b)
        self.assertEqual(a[2], a[5][:2])
        self.assertEqual(a[5], a[10][:5])

    def test_invalid_budget(self):
        for n in [0, -1, 13, True]:
            with self.assertRaises(ValueError):
                nested_subsets(self.splits["train"], [n], 42)

    def test_regime_views_and_target_isolation(self):
        e = self.splits["test"][0]
        for form in ["answer_only", "trajectory_supervised", "experiential"]:
            view = e.view(form)
            self.assertNotIn("answer", view)
            self.assertNotIn("trajectory", view)
            self.assertNotIn("metadata", view)
            view["observation"]["operands"][0] = 999
            self.assertNotEqual(e.observation["operands"][0], 999)
        self.assertIn("answer", e.view("answer_only", True))
        self.assertIn("trajectory", e.view("trajectory_supervised", True))
        self.assertNotIn("trajectory", e.view("experiential", True))
        self.assertNotIn("answer", e.view("experiential", True))

    def test_roundtrip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data"
            save_dataset(path, self.splits, self.config)
            loaded, _ = load_dataset(path)
            self.assertEqual(stable_hash({k: [e.to_dict() for e in v] for k, v in loaded.items()}),
                             stable_hash({k: [e.to_dict() for e in v] for k, v in self.splits.items()}))
            with self.assertRaises(FileExistsError):
                save_dataset(path, self.splits, self.config)

    def test_byte_reproducibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a", Path(tmp) / "b"
            save_dataset(a, generate(config()), config())
            save_dataset(b, generate(config()), config())
            for file in a.iterdir():
                self.assertEqual(file.read_bytes(), (b / file.name).read_bytes())

    def test_checksum_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data"
            save_dataset(path, self.splits, self.config)
            (path / "test.jsonl").write_text("{}\n")
            with self.assertRaises(ValueError):
                load_dataset(path)

    def test_reference_metrics(self):
        e = self.splits["test"][0]
        r = evaluate(e, ReferenceModel().predict(e.view("experiential")), self.backend)
        metrics = aggregate([r])
        for key in ["accuracy", "trajectory_validity", "grounded_rate", "goal_success_rate"]:
            self.assertEqual(metrics[key], 1)
        self.assertEqual(metrics["unsupported_rate"], 0)

    def test_answer_only_unsupported(self):
        e = self.splits["test"][0]
        r = evaluate(e, Prediction(e.target_answer), self.backend)
        metrics = aggregate([r])
        self.assertEqual(metrics["accuracy"], 1)
        self.assertEqual(metrics["grounded_rate"], 0)
        self.assertEqual(metrics["goal_success_rate"], 0)
        self.assertIsNone(metrics["trajectory_validity"])
        self.assertEqual(r.error_type, "unsupported_answer")

    def test_wrong_and_malformed_predictions(self):
        e = self.splits["test"][0]
        correct = ReferenceModel().predict(e.view("experiential"))
        wrong = evaluate(e, replace(correct, answer=-999), self.backend)
        self.assertFalse(wrong.correct)
        self.assertFalse(wrong.goal_reached)
        self.assertFalse(wrong.grounded)
        self.assertTrue(wrong.trajectory_valid)
        malformed = evaluate(e, Prediction(e.target_answer, {}), self.backend)
        self.assertEqual(malformed.error_type, "malformed_trajectory")
        self.assertFalse(malformed.grounded)

    def test_foreign_trajectory_rejected(self):
        e, other = self.splits["test"][:2]
        foreign = ReferenceModel().predict(other.view("experiential"))
        r = evaluate(e, replace(foreign, answer=e.target_answer), self.backend)
        self.assertFalse(r.trajectory_valid)

    def test_incomplete_trajectory(self):
        e = self.splits["test"][0]
        pred = ReferenceModel().predict(e.view("experiential"))
        pred.trajectory["transitions"] = pred.trajectory["transitions"][:-1]
        r = evaluate(e, pred, self.backend)
        self.assertTrue(r.trajectory_valid)
        self.assertFalse(r.goal_reached)
        self.assertFalse(r.grounded)

    def test_metric_denominators(self):
        e = self.splits["test"][0]
        good = evaluate(e, ReferenceModel().predict(e.view("experiential")), self.backend)
        missing = evaluate(e, Prediction(), self.backend)
        metrics = aggregate([good, missing])
        self.assertEqual(metrics["accuracy"], .5)
        self.assertEqual(metrics["grounded_rate"], .5)
        self.assertEqual(metrics["goal_success_rate"], .5)
        self.assertEqual(metrics["trajectory_validity"], 1)
        self.assertEqual(metrics["unsupported_rate"], .5)
        self.assertIsNone(aggregate([])["accuracy"])

    def test_n_star_der(self):
        self.assertEqual(n_star({10: .7, 25: .91, 50: .95}, .9), 25)
        self.assertEqual(n_star({10: .7}, .9), "not reached")
        self.assertEqual(data_efficiency_ratio({100: .9}, {25: .9}, .9), 4)
        self.assertEqual(data_efficiency_ratio({100: .8}, {25: .9}, .9), "not reached")
        for curve, alpha in [({0: .9}, .9), ({10: float("nan")}, .9), ({10: 2}, .9), ({10: .9}, 2)]:
            with self.assertRaises(ValueError):
                n_star(curve, alpha)

    def test_runner_artifacts_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data"
            save_dataset(path, self.splits, self.config)
            a, ma = run(ReferenceModel(), path, Path(tmp) / "results")
            b, mb = run(ReferenceModel(), path, Path(tmp) / "results")
            self.assertNotEqual(a, b)
            for name in ["config.json", "metrics.json", "predictions.jsonl", "summary.md"]:
                self.assertTrue((a / name).exists())
            for key in ["accuracy", "grounded_rate", "trajectory_validity", "goal_success_rate"]:
                self.assertEqual(ma[key], mb[key])
            self.assertEqual((a / "predictions.jsonl").read_bytes(), (b / "predictions.jsonl").read_bytes())
            _, ood = run(ReferenceModel(), path, Path(tmp) / "results", "ood_test", training_size=5)
            self.assertEqual(ood["ood_accuracy"], 1)
            self.assertIsNone(ood["training_time"])

    def test_runner_validates_before_model_calls(self):
        class NeverCalled:
            identifier = "never"
            def predict(self, example):
                raise AssertionError("Must not be called")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data"
            save_dataset(path, self.splits, self.config)
            (path / "train.jsonl").write_text("{}\n")
            with self.assertRaises(ValueError):
                run(NeverCalled(), path, Path(tmp) / "results")

    def test_runner_records_model_failure(self):
        class Failing:
            identifier = "failing"
            parameter_count = None
            def predict(self, example):
                raise TimeoutError()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data"
            save_dataset(path, self.splits, self.config)
            folder, metrics = run(Failing(), path, Path(tmp) / "results")
            self.assertEqual(metrics["accuracy"], 0)
            rows = [json.loads(line) for line in (folder / "predictions.jsonl").read_text().splitlines()]
            self.assertTrue(all(r["error_type"] == "timeout" for r in rows))

    def test_non_arithmetic_backend_reuses_pipeline(self):
        from mindscape.data.backends import REGISTRY, register
        from mindscape.data.schemas import BenchmarkExample
        class ToyBackend:
            name = "toy_test_only"
            version = "test-1"
            def structures(self, spec): return spec["structures"]
            def structure_identity(self, category): return category
            def sample(self, category, rng, config): return {"category": category, "value": rng.randrange(1000000)}
            def identity(self, observation): return observation["category"], observation["value"]
            def category(self, observation): return observation["category"]
            def example(self, observation, seed, split):
                return BenchmarkExample(stable_hash(observation), self.name, str(observation), observation,
                                        {}, {}, observation["value"], [{"evidence": observation}],
                                        {"random_seed": seed, "split": split})
            def validate(self, example): pass
            def assess(self, example, answer, trajectory): return True, answer == example.target_answer, "correct"
            def reference(self, view): return view["observation"]["value"], {"trace": view["observation"]}
        register(ToyBackend())
        try:
            cfg = {"environment": ToyBackend.name, "seed": 42, "splits": {
                "train": {"count": 4, "structures": ["alpha"]},
                "validation": {"count": 2, "structures": ["alpha"]},
                "test": {"count": 2, "structures": ["alpha"]},
                "ood_test": {"count": 2, "structures": ["beta"]}}}
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "data"
                save_dataset(path, generate(cfg), cfg)
                _, metrics = run(ReferenceModel(), path, Path(tmp) / "results", "ood_test")
                self.assertEqual(metrics["accuracy"], 1)
        finally:
            del REGISTRY[ToyBackend.name]

    def test_seed_metadata_tamper(self):
        splits = copy.deepcopy(self.splits)
        e = splits["test"][0]
        splits["test"][0] = self.backend.example(e.observation, 99, "test")
        with self.assertRaises(ValueError):
            validate_dataset(splits, self.config)

    def test_measurable_difficulty_filters(self):
        cfg = config()
        cfg["difficulty_bounds"] = {"carry_count": {"min": 1}}
        splits = generate(cfg)
        self.assertTrue(all(e.metadata["difficulty"]["carry_count"] >= 1
                            for rows in splits.values() for e in rows))

    def test_signed_zero_configuration(self):
        cfg = config()
        cfg.update(signed=True, include_zero=True)
        splits = generate(cfg)
        self.assertTrue(validate_dataset(splits, cfg))
        self.assertTrue(any(x < 0 for rows in splits.values() for e in rows
                            for x in e.observation["operands"]))
