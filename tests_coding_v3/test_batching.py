"""Accounting/transport tests for the throughput-only wrapper."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path("scripts/v3_coding").resolve()))
spec = importlib.util.spec_from_file_location(
    "coding_batch_study", "scripts/v3_coding/batched_study.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class BatchTests(unittest.TestCase):
    def test_client_preserves_frozen_prompt_and_token_cap(self):
        class Client:
            def __init__(self):
                self.timings = [{"tokens_in": 2, "tokens_out": 3}]

            def generate(self, *args):
                self.args = args
                return "response"

        client = Client()
        payload = {"repository": {"solution.py": "pass"}, "visible_tests": {}}
        text, timing = module.ClientBackend(client).generate(payload, 256)
        self.assertEqual(text, "response")
        self.assertEqual(client.args, (module.SYSTEM, json.dumps(payload), 256, 11))
        self.assertEqual(timing["tokens_out"], 3)

    def test_duplicate_requests_count_one_actual_generation(self):
        class Tokenizer:
            eos_token_id = 9

            def apply_chat_template(self, *args, **kwargs):
                return "request"

            def __call__(self, texts, **kwargs):
                return {
                    "input_ids": torch.ones((len(texts), 2), dtype=torch.int64),
                    "attention_mask": torch.ones((len(texts), 2), dtype=torch.int64),
                }

            def decode(self, ids, **kwargs):
                return "generated"

        class Model:
            def generate(self, **kwargs):
                inp = kwargs["input_ids"]
                return torch.cat((inp, torch.tensor([[4, 9]] * len(inp))), dim=1)

        with tempfile.TemporaryDirectory() as d:
            coder = object.__new__(module.CpuBatchCoder)
            coder.cache = Path(d)
            coder.model_hash = "test"
            coder.tokenizer = Tokenizer()
            coder.model = Model()
            coder.calls = 0
            coder.input_tokens = 0
            coder.output_tokens = 0
            coder.batch_records = []
            coder.deadline = None
            request = ("system", "{}", 16, 11)
            result = coder.generate_batch([request, request])
            self.assertEqual(coder.calls, 1)
            self.assertEqual(coder.output_tokens, 2)
            self.assertFalse(result[0][1]["cache_hit"])
            self.assertTrue(result[1][1]["cache_hit"])
            result = coder.generate_batch([request])
            self.assertTrue(result[0][1]["cache_hit"])
            self.assertEqual(coder.calls, 1)

    def test_corrupt_cached_response_is_rejected(self):
        from mindscape.v3.schema import digest

        with tempfile.TemporaryDirectory() as d:
            coder = object.__new__(module.CpuBatchCoder)
            coder.cache = Path(d)
            coder.model_hash = "test"
            request = ("system", "{}", 16, 11)
            key = digest(["test", list(request), "cpu-float32-batched-v1"])
            (Path(d) / (key + ".json")).write_text(
                json.dumps({"response": "tamper", "response_hash": "wrong"})
            )
            with self.assertRaises(ValueError):
                coder.generate_batch([request])


class PathSchemaTests(unittest.TestCase):
    def test_schema_remediation_does_not_modify_response_or_code(self):
        spec = importlib.util.spec_from_file_location(
            "path_schema", "scripts/v3_coding/path_schema.py"
        )
        wrapper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wrapper)

        class Backend:
            def generate(self, payload, max_tokens):
                self.payload = payload
                return "unaltered-model-response", {"tokens_out": 3}

        backend = Backend()
        original = {
            "repository": {"rules.py": "pass", "solution.py": "pass"},
            "instruction": "old example existing_file.py",
        }
        result = wrapper.PathSchemaBackend(backend).generate(original, 256)
        self.assertEqual(result[0], "unaltered-model-response")
        self.assertEqual(result[1]["actual_model_payload"], backend.payload)
        self.assertEqual(backend.payload["allowed_edit_paths"], ["rules.py", "solution.py"])
        self.assertNotIn("existing_file.py", backend.payload["instruction"])
        self.assertIn("existing_file.py", original["instruction"])


class CoordinatorTests(unittest.TestCase):
    def test_all_independent_jobs_complete(self):
        from cpu_coordinator import evaluate_batch

        class Coder:
            revision = "fake"

            def generate_batch(self, requests):
                return [("response", {"tokens_out": 1}) for _ in requests]

        def solve(job, client):
            return {"job": job, "response": client.generate("s", "p", 8, 11)}

        saved = []
        self.assertEqual(evaluate_batch(Coder(), list(range(4)), solve, saved.append, 2), 4)
        self.assertEqual({r["job"] for r in saved}, set(range(4)))

    def test_interrupt_releases_waiting_clients(self):
        from cpu_coordinator import evaluate_batch

        class Coder:
            revision = "fake"

            def generate_batch(self, requests):
                raise KeyboardInterrupt()

        def solve(job, client):
            return client.generate("s", "p", 8, 11)

        import time

        started = time.perf_counter()
        with self.assertRaises(KeyboardInterrupt):
            evaluate_batch(Coder(), list(range(8)), solve, lambda x: None, 2)
        self.assertLess(time.perf_counter() - started, 2)

    def test_expired_deadline_performs_no_model_work(self):
        from cpu_coordinator import evaluate_batch

        class Coder:
            revision = "fake"

            def generate_batch(self, requests):
                raise AssertionError("Deadline ignored")

        def solve(job, client):
            return client.generate("s", "p", 8, 11)

        self.assertEqual(evaluate_batch(Coder(), [1, 2], solve, lambda x: None, 2, deadline=0), 0)
