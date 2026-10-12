import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from mindscape.v3.coding_learning.agent import solve
from mindscape.v3.coding_learning.backend import parse
from mindscape.v3.coding_learning.concepts import extract, similarity
from mindscape.v3.coding_learning.data import load_public, make_task
from mindscape.v3.coding_learning.execution import Executor, safe_source
from mindscape.v3.coding_learning.memory import VerifiedMemory
from mindscape.v3.coding_learning.planning import ProposalEvaluator, choose, features, fit
from mindscape.v3.coding_learning.schema import RepairProposal
from mindscape.v3.schema import Concept, Relationship


class CodingLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.executor = Executor("work/coding/runtime")

    def setUp(self):
        self.task, self.secret = make_task("train", "threshold", 31, 7)

    def test_public_boundary(self):
        text = json.dumps(self.task.visible())
        self.assertNotIn("teacher_repository", text)
        self.assertNotIn("private_tests", text)
        self.assertNotIn("ground_truth_patch", text)

    def test_manifest_identity_and_split_isolation(self):
        tasks, m = load_public("configs/v3_coding/benchmark_v1")
        ids = {t.identity for t in tasks}
        self.assertEqual(len(ids), 48)
        self.assertEqual(m["counts"], {"train": 24, "validation": 6, "iid": 6, "ood": 12})

    def test_generated_all_families_executed(self):
        root = Path("configs/v3_coding/benchmark_v1")
        rows = json.loads((root / "validation.json").read_text())
        self.assertEqual(len(rows), 48)
        for row in rows:
            for key in ("broken_visible", "broken_private"):
                self.assertFalse(row["executions"][key]["all_passed"])
            for key in ("teacher_visible", "teacher_private"):
                self.assertTrue(row["executions"][key]["all_passed"])

    def test_ast_concepts_common_schema(self):
        result = extract(self.task.repository, self.task.contract)
        c = result["concepts"][0]
        self.assertEqual(c["domain"], "python_repair")
        rebuilt = Concept(
            c["id"],
            c["domain"],
            c["description"],
            tuple(Relationship(**r) for r in c["relationships"]),
        )
        self.assertEqual(rebuilt.domain, "python_repair")
        self.assertTrue(result["relationships"])

    def test_similarity_renaming_invariance(self):
        a = extract(self.task.repository, self.task.contract)
        b = extract(
            {
                k: v.replace("values", "renamed_vector").replace("item", "renamed_item")
                for k, v in self.task.repository.items()
            },
            self.task.contract,
        )
        self.assertAlmostEqual(similarity(a, b), 1, places=5)

    def test_ast_does_not_execute(self):
        marker = Path("work/should-not-exist-v3")
        extract(
            {"x.py": "open('work/should-not-exist-v3','w').write('unsafe')"}, self.task.contract
        )
        self.assertFalse(marker.exists())

    def test_paths_and_new_files_rejected(self):
        for path in ("../bad.py", "/tmp/bad.py", "new.py"):
            with self.assertRaises(ValueError):
                RepairProposal({path: "pass"}).apply(self.task.repository)

    def test_dynamic_and_external_capabilities_rejected(self):
        for source in (
            "import os",
            'eval("1")',
            "x.__class__",
            "import sys",
            "from subprocess import run",
        ):
            with self.assertRaises(ValueError):
                safe_source({"solution.py": source})

    def test_parser_bounded_and_exact(self):
        p = parse(
            json.dumps({"candidates": [{"repository": self.task.repository}]}), self.task.repository
        )
        self.assertEqual(len(p), 1)
        for value in (
            {"candidates": []},
            {"private_tests": []},
            {"candidates": [{"repository": {"missing.py": "pass"}}]},
        ):
            with self.assertRaises(ValueError):
                parse(json.dumps(value), self.task.repository)

    def test_real_wasi_teacher_and_defect(self):
        self.assertFalse(
            self.executor.run(self.task.repository, self.task.visible_tests)["all_passed"]
        )
        self.assertTrue(
            self.executor.run(self.secret["teacher_repository"], self.task.visible_tests)[
                "all_passed"
            ]
        )

    def test_real_cross_file(self):
        t, s = make_task("ood", "cross_file", 21, 8)
        self.assertTrue(
            self.executor.run(s["teacher_repository"], s["private_tests"])["all_passed"]
        )
        self.assertFalse(self.executor.run(t.repository, s["private_tests"])["all_passed"])

    def test_real_mutation_contract(self):
        t, _s = make_task("ood", "alias_mutation", 21, 8)
        r = self.executor.run(t.repository, t.visible_tests)
        self.assertFalse(r["all_passed"])
        self.assertEqual(r["observed"][0]["value"], [1, 2, 3])
        self.assertNotEqual(r["observed"][0]["args_after"], t.visible_tests["cases"][0]["args"])

    def test_memory_rejects_eval_and_failure(self):
        m = VerifiedMemory([self.task])
        with self.assertRaises(ValueError):
            m.admit(
                replace(self.task, split="ood"),
                self.secret["teacher_repository"],
                self.executor,
                self.secret["private_tests"],
            )
        with self.assertRaises(ValueError):
            m.admit(self.task, self.task.repository, self.executor, self.secret["private_tests"])
        self.assertEqual(m.records, [])

    def test_memory_real_success_roundtrip_and_tamper(self):
        m = VerifiedMemory([self.task])
        m.admit(
            self.task,
            self.secret["teacher_repository"],
            self.executor,
            self.secret["private_tests"],
        )
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "memory.json"
            m.save(path)
            restored = VerifiedMemory.load(
                path, [self.task], self.executor, {self.task.identity: self.secret}
            )
            self.assertEqual(restored.records[0]["after"], self.secret["teacher_repository"])
            rows = json.loads(path.read_text())
            rows[0]["task_id"] = "bad"
            path.write_text(json.dumps(rows))
            with self.assertRaises(ValueError):
                VerifiedMemory.load(
                    path, [self.task], self.executor, {self.task.identity: self.secret}
                )

    def test_evaluator_training_gradient_and_determinism(self):
        p = RepairProposal(self.secret["teacher_repository"])
        x = features(self.task.repository, p, self.task.contract)
        rows = [{"split": "train", "features": x.tolist(), "visible_pass": True, "seconds": 0.2}]
        a, _r = fit(rows, 11, 10)
        b, _ = fit(rows, 11, 10)
        self.assertTrue(all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items()))
        initial = ProposalEvaluator(11)
        self.assertTrue(
            any(not torch.equal(v, initial.state_dict()[k]) for k, v in a.state_dict().items())
        )
        self.assertIsNone(a.predict(x)["private_success"])
        with self.assertRaises(ValueError):
            fit([dict(rows[0], split="ood")], 11)

    def test_no_private_feature_channel(self):
        p = RepairProposal(self.secret["teacher_repository"])
        x = features(self.task.repository, p, self.task.contract)
        self.assertTrue(np.isfinite(x).all())
        self.assertEqual(x.shape, (112,))

    def test_zero_ranker_cannot_supply_correct_patch(self):
        model = ProposalEvaluator()
        [p.data.zero_() for p in model.parameters()]
        proposals = [
            RepairProposal(self.task.repository),
            RepairProposal(self.secret["teacher_repository"]),
        ]
        index, _ = choose(self.task.repository, proposals, self.task.contract, [], model)
        self.assertEqual(index, 0)
        self.assertFalse(
            self.executor.run(
                proposals[index].apply(self.task.repository), self.task.visible_tests
            )["all_passed"]
        )

    def test_loop_executes_visible_and_keeps_eval_memory_immutable(self):
        task = replace(self.task, split="ood")
        secret = self.secret
        seen = []

        class Backend:
            def generate(self, payload, max_tokens):
                seen.append(payload)
                return json.dumps(
                    {
                        "hypothesis": "boundary",
                        "candidates": [{"repository": secret["teacher_repository"]}],
                    }
                ), {"tokens_in": 1, "tokens_out": 1}

        m = VerifiedMemory([self.task])
        before = copy.deepcopy(m.records)
        result = solve(task, Backend(), self.executor, m, ProposalEvaluator(), "mindscape")
        self.assertIsNone(result["success"])
        self.assertEqual(m.records, before)
        self.assertTrue(result["attempts"][0]["actual_visible_result"]["all_passed"])
        self.assertNotIn("private_tests", json.dumps(seen))
        self.assertNotIn("teacher_repository", json.dumps(seen))

    def test_capability_isolation_probe(self):
        self.assertTrue(all(self.executor.sandbox.probe().values()))


if __name__ == "__main__":
    unittest.main()
