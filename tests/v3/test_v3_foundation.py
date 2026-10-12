"""Foundation scientific/security controls; standard-library unittest for CPU portability."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from mindscape.v3.baselines import TrainingBudget, require_fair
from mindscape.v3.domains import DOMAINS, Task, generate_tasks, interpret_python
from mindscape.v3.experiment import (
    freeze_manifest,
    lock,
    read_manifest,
    run,
    validate_tasks,
    wilson,
)
from mindscape.v3.kernel import BoundedPolicy, Kernel, replay
from mindscape.v3.memory import Memory
from mindscape.v3.models import HistoricalBackend, TinyMLP
from mindscape.v3.schema import (
    Action,
    Concept,
    Experience,
    Goal,
    Observation,
    Prediction,
    Provenance,
    Rule,
    State,
    decode,
    encode,
)


class SchemaTests(unittest.TestCase):
    def test_roundtrip_all_nested_evidence(self):
        domain = DOMAINS["mathematics"]
        task = Task("mathematics", (2, 3), "train", "unit")
        before = State(domain.observe(task))
        transition = domain.execute(before, Action("submit", 4))
        exp = Experience(
            transition,
            domain.evaluate(task, transition),
            Provenance(task.identity, "train", "model", "source", 0),
        )
        for record in (
            exp,
            Prediction(Action("submit", 4), before, 0.5),
            Concept("sum", "mathematics", "addition"),
            Rule("sum", "hypothesis", 0.2, ("ref",)),
        ):
            self.assertEqual(decode(encode(record)), record)

    def test_reject_codecs_unknown_fields_nan_and_wrong_types(self):
        for payload in (
            '{"schema":"pickle"}',
            ('{"schema":"mindscape.v3/1","type":"os.system","data":{}}'),
        ):
            with self.assertRaises(ValueError):
                decode(payload)
        with self.assertRaises(ValueError):
            Observation("math", (float("nan"),), Goal("sum"))
        with self.assertRaises(ValueError):
            Observation("math", (), Goal("sum"), missing=("x",))
        payload = json.loads(encode(Action("submit", 1)))
        payload["data"]["value"] = True
        with self.assertRaises(ValueError):
            decode(json.dumps(payload))
        with self.assertRaises(ValueError):
            Prediction(Action("submit", 1), State(Observation("m", (), Goal("g"))), 1.2)


class ScientificControls(unittest.TestCase):
    def test_candidate_set_has_no_semantic_mask(self):
        for domain in DOMAINS.values():
            tasks = [t for t in generate_tasks() if t.domain == domain.name]
            candidate_sets = [domain.candidates(State(domain.observe(t))) for t in tasks]
            self.assertTrue(all(c == candidate_sets[0] for c in candidate_sets))
            self.assertEqual(len(candidate_sets[0]), 65)

    def test_zero_and_random_cannot_be_perfect_by_controller(self):
        for zero in (True, False):
            for seed in (0, 1, 2):
                for domain in DOMAINS.values():
                    tasks = [t for t in generate_tasks() if t.domain == domain.name]
                    results = [
                        Kernel(TinyMLP(seed, zero)).run(t, domain, seed, "unit") for t in tasks
                    ]
                    self.assertLess(sum(r["success"] for r in results), len(results))
                    self.assertTrue(all(r["candidate_count"] == 65 for r in results))

    def test_wrong_proposal_is_executed_unchanged(self):
        d = DOMAINS["mathematics"]
        task = Task("mathematics", (2, 3), "iid", "unit")
        trace = Kernel(TinyMLP(0, True)).run(task, d, 0, "unit")
        self.assertEqual(trace["events"][2]["value"]["action"]["value"], -32)
        self.assertEqual(trace["events"][4]["value"]["after"]["written"], (-32.0,))
        self.assertFalse(trace["success"])
        self.assertTrue(trace["valid"])
        self.assertFalse(replay(trace, d).success)

    def test_verifier_called_after_execution_only(self):
        wrapped = DOMAINS["dynamics"]

        class Spy:
            acted = False

            def observe(self, task):
                return wrapped.observe(task)

            def candidates(self, state):
                return wrapped.candidates(state)

            def execute(self, state, action):
                self.acted = True
                return wrapped.execute(state, action)

            def evaluate(self, task, transition):
                if not self.acted:
                    raise AssertionError("Premature verifier")
                return wrapped.evaluate(task, transition)

        Kernel(TinyMLP()).run(Task("dynamics", (2, 3, 1), "iid", "unit"), Spy(), 0, "unit")

    def test_policy_rejects_singleton_and_nonfinite(self):
        policy = BoundedPolicy()
        for scores, actions in (
            ([1], (Action("s", 1),)),
            ([float("nan"), 0], (Action("s", 1), Action("s", 2))),
            ([1, 2], (Action("s", 1), Action("s", 1))),
        ):
            with self.assertRaises(ValueError):
                policy.choose(scores, actions)

    def test_deterministic_and_replay_tamper(self):
        task = Task("dynamics", (2, 3, 1), "iid", "unit")
        d = DOMAINS["dynamics"]
        a = Kernel(TinyMLP(2)).run(task, d, 2, "unit")
        b = Kernel(TinyMLP(2)).run(task, d, 2, "unit")
        self.assertEqual(a["events"], b["events"])
        a["events"][2]["value"]["action"]["value"] += 1
        with self.assertRaises(ValueError):
            replay(a, d)

    def test_independent_expected_results(self):
        for name, inputs, answer in (
            ("mathematics", (3, 4), 7),
            ("dynamics", (3, 4, 2), 9),
            ("python", (3, 4, 2), 2),
        ):
            task = Task(name, inputs, "iid", "unit")
            d = DOMAINS[name]
            state = State(d.observe(task))
            self.assertTrue(
                d.evaluate(task, d.execute(state, d.candidates(state)[answer + 32])).success
            )
            other = replace(task, inputs=tuple(x + 1 for x in inputs))
            self.assertFalse(
                d.evaluate(other, d.execute(state, d.candidates(state)[answer + 32])).valid
            )


class SecurityAndMemory(unittest.TestCase):
    def test_python_capability_subset(self):
        self.assertEqual(interpret_python("x * 3 + 2", 4), 14)
        for source in (
            "__import__('os').system('id')",
            'open("/etc/passwd")',
            "x.__class__",
            "2**1000000",
            "[x for x in range(10)]",
            "(lambda: 1)()",
            "99999999999999",
            "True",
        ):
            with self.assertRaises((ValueError, SyntaxError)):
                interpret_python(source, 2)
        with self.assertRaises(ValueError):
            interpret_python("x+" * 300 + "1", 2)

    def test_memory_partition_and_hypothesis_boundary(self):
        memory = Memory()
        d = DOMAINS["mathematics"]
        task = Task("mathematics", (2, 3), "train", "unit")
        transition = d.execute(State(d.observe(task)), Action("submit", 5))
        exp = Experience(
            transition,
            d.evaluate(task, transition),
            Provenance(task.identity, "train", "m", "s", 0),
        )
        key = memory.add(exp)
        self.assertEqual(memory.retrieve("mathematics", "train"), (exp,))
        self.assertEqual(memory.retrieve("mathematics", "train", exclude_task=task.identity), ())
        with self.assertRaises(ValueError):
            memory.retrieve("mathematics", "iid")
        with self.assertRaises(ValueError):
            memory.add(Prediction(Action("submit", 1), transition.after, 0.5))
        memory.store_concept(Concept("sum", "mathematics", "sum"))
        memory.hypothesize(Rule("sum", "possible", 0.2, (key,)))
        with self.assertRaises(ValueError):
            memory.hypothesize(Rule("sum", "x", 0.5, ("fake",)))
        heldout = memory.add(replace(exp, provenance=replace(exp.provenance, split="iid")))
        with self.assertRaises(ValueError):
            memory.hypothesize(Rule("sum", "x", 0.5, (heldout,)))

    def test_eval_does_not_update_training_memory(self):
        k = Kernel(TinyMLP())
        k.run(Task("mathematics", (2, 3), "iid", "unit"), DOMAINS["mathematics"], 0, "unit")
        self.assertFalse(k.memory.episodes)

    def test_historical_adapter_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                HistoricalBackend(tmp, {})
            Path(tmp, "config.json").write_text('{"model_type":"custom"}')
            import hashlib

            hashes = {
                "config.json": hashlib.sha256(Path(tmp, "config.json").read_bytes()).hexdigest()
            }
            with self.assertRaises(ValueError):
                HistoricalBackend(tmp, hashes)
            with self.assertRaises(ValueError):
                HistoricalBackend(tmp, {"../outside": "x"})


class ExperimentTests(unittest.TestCase):
    def test_split_identity_leakage_and_fairness(self):
        tasks = generate_tasks()
        validate_tasks(tasks)
        with self.assertRaises(ValueError):
            validate_tasks(tasks + (replace(tasks[0], split="iid"),))
        budget = TrainingBudget("hash", ("a",), 1, 1, 0)
        require_fair(budget, budget)
        with self.assertRaises(ValueError):
            require_fair(budget, replace(budget, verifier_queries=1))

    def test_checkpoint_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "model.json")
            model = TinyMLP(3)
            model.save(path)
            self.assertEqual(
                model.checkpoint(), TinyMLP.load(path, model.checkpoint_hash).checkpoint()
            )
            data = json.loads(path.read_text())
            data["weights"][0] += 1
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                TinyMLP.load(path, model.checkpoint_hash)

    def test_freeze_resume_integrity_and_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, output = Path(tmp, "manifest.json"), Path(tmp, "run")
            freeze_manifest(manifest)
            with self.assertRaises(FileExistsError):
                freeze_manifest(manifest)
            first = run(manifest, output, seeds=(0,))
            second = run(manifest, output, seeds=(0,))
            self.assertEqual(first["completed"], 150)
            self.assertEqual(second["resumed"], 150)
            self.assertEqual(first["rows"], second["rows"])
            with self.assertRaises(ValueError):
                run(manifest, output, seeds=(1,))
            trace = next((output / "traces").glob("*.json"))
            data = json.loads(trace.read_text())
            data["trace"]["success"] = not data["trace"]["success"]
            trace.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                run(manifest, output, seeds=(0,))

    def test_manifest_tamper_lock_and_intervals(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "manifest.json")
            freeze_manifest(path)
            data = json.loads(path.read_text())
            data["body"]["seed"] += 1
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                read_manifest(path)
            with lock(Path(tmp)), self.assertRaises(FileExistsError), lock(Path(tmp)):
                pass
        lo, hi = wilson(0, 10)
        self.assertAlmostEqual(lo, 0)
        self.assertGreater(hi, 0)
        lo, hi = wilson(10, 10)
        self.assertLess(lo, 1)
        self.assertAlmostEqual(hi, 1)
        with self.assertRaises(ValueError):
            wilson(0, 0)


if __name__ == "__main__":
    unittest.main()
