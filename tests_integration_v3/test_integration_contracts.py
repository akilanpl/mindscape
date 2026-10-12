import json
import unittest
from pathlib import Path

from mindscape.v3.coding_learning.execution import safe_source
from mindscape.v3.integration import Request, Response, Runtime
from mindscape.v3.integration.runtime import Adapter
from mindscape.v3.kernel import Kernel
from mindscape.v3.schema import Action, Goal, Observation, Prediction, State, decode, encode


class Contracts(unittest.TestCase):
    def test_observed_math_claims_match_preserved_scaffold(self):
        from mindscape.v3.math_learning.environment import Claim, Problem, run_problem

        paths = list(Path("experiments/v3_integration/native_v1").glob("*-mathematics.json"))
        self.assertEqual(len(paths), 2)
        for path in paths:
            saved = json.loads(path.read_text())
            proposal = saved["unified"]["prediction"]
            steps = iter(proposal["steps"])

            class SavedClaims:
                def predict_local(self, state, steps=steps, case=self):
                    from dataclasses import asdict

                    step = next(steps)
                    case.assertEqual(asdict(state), step["state"])
                    return Claim(**step["claim"]), {"consistency": 0.5}

                def predict_sign(self, visible, answer=proposal["answer"]):
                    return int(answer.startswith("-")), {"kind": "saved_claim_replay"}

            inputs = saved["request"]["inputs"]
            problem = Problem(
                tuple(inputs["operands"]), tuple(inputs["operations"]), "unit", "integration"
            )
            replayed = run_problem(problem, SavedClaims())
            self.assertEqual(str(replayed["answer"]), proposal["answer"])
            self.assertFalse(replayed["success"])
            self.assertIsNone(next(steps, None))

    def test_bridge_real_kernel_all_domains(self):
        # Non-neural contract fixture: no scientific performance is inferred here.
        class ContractAdapter(Adapter):
            def observation(self):
                return self.request.inputs

            def propose(self, public):
                return [{"summary": [1.0]}, {"summary": [0.0]}], [1.0, 0.0]

            def act(self, proposal):
                return {"summary": proposal["summary"]}

            def verify(self, actual):
                return {
                    "valid": True,
                    "success": True,
                    "evaluator": "contract-fixture",
                    "reason": "test",
                }

        records = json.loads(Path("configs/v3_integration/requests.json").read_text())["requests"]
        for record in records:
            with self.subTest(domain=record["domain"]):
                request = Request(**record)
                adapter = ContractAdapter(request, {"parameter_counts": {"fixture": 17}})
                backend = adapter.backend
                self.assertEqual(backend.identifier, "v3.integration." + request.domain)
                result = Kernel(backend).run(request, adapter, 11, "contract-test")
                serialized = json.loads(json.dumps(result, allow_nan=False))
                self.assertEqual(serialized["parameter_count"], 17)
                self.assertEqual(serialized["model_calls"], 0)
                self.assertEqual(len(serialized["events"]), 7)
                self.assertEqual(serialized["events"][-1]["value"]["training_memory_key"], None)
                self.assertEqual(backend.checkpoint(), adapter.provenance)

    def test_bridge_rejects_unmeasured_parameter_counts(self):
        request = Request("mathematics", {"operands": [7, 8], "operations": ["add"]})
        for counts in ({}, {"model": 0}, {"model": None}):
            with self.assertRaises(ValueError):
                _ = Adapter(request, {"parameter_counts": counts}).backend.parameter_count

    def test_three_domains_frozen_requests(self):
        data = json.loads(Path("configs/v3_integration/requests.json").read_text())
        self.assertEqual(
            {Request(**r).domain for r in data["requests"]}, {"mathematics", "coding", "physics"}
        )

    def test_private_fields_rejected(self):
        with self.assertRaises(ValueError):
            Request("mathematics", {"operands": [7, 8], "operations": ["add"], "expected": 15})

    def test_inputs_copied(self):
        inputs = {"operands": [7, 8], "operations": ["add"]}
        request = Request("mathematics", inputs)
        inputs["operands"][0] = 99
        self.assertEqual(request.inputs["operands"][0], 7)

    def test_mutated_request_revalidated_before_model(self):
        request = Request("mathematics", {"operands": [7, 8], "operations": ["add"]})
        request.inputs["expected"] = 15
        with self.assertRaises(ValueError):
            Runtime({}, "test").run(request)

    def test_invalid_numeric_math(self):
        with self.assertRaises(ValueError):
            Request("mathematics", {"operands": [True, 8], "operations": ["add"]})

    def test_nonfinite_physics(self):
        with self.assertRaises(ValueError):
            Request(
                "physics",
                {
                    "family": "motion",
                    "seed": 1,
                    "index": 0,
                    "horizon": 8,
                    "target": [float("nan"), 0],
                },
            )

    def test_bounded_physics(self):
        with self.assertRaises(ValueError):
            Request(
                "physics",
                {"family": "motion", "seed": 1, "index": 0, "horizon": 100, "target": [0, 0]},
            )

    def test_unknown_domain(self):
        with self.assertRaises(ValueError):
            Request("calculator", {})

    def test_response_requires_real_outcome_type(self):
        with self.assertRaises(ValueError):
            Response(
                "id", "mathematics", {"kind": "hypothetical"}, {"kind": "hypothetical"}, {}, {}
            )

    def test_existing_kernel_codec_prediction(self):
        p = Prediction(
            Action("submit_proposal", 0),
            State(Observation("physics", (), Goal("goal")), (1.0, 2.0), True),
            0.5,
        )
        self.assertEqual(decode(encode(p)), p)

    def test_unsafe_generated_code_rejected(self):
        for source in [
            "import os\ndef f(): return os.getcwd()",
            "def f(): return eval('1')",
            "def f(): return open('/etc/passwd')",
        ]:
            with self.assertRaises(ValueError):
                safe_source({"solution.py": source})

    def test_no_private_suite_in_public_requests(self):
        public = json.loads(Path("configs/v3_integration/requests.json").read_text())
        private = json.loads(Path("configs/v3_integration/private.json").read_text())
        self.assertEqual(len(private), 1)
        for record in public["requests"]:
            self.assertNotIn("private_tests", record["inputs"])
