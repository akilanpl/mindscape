"""Accounting/transport tests for the throughput-only wrapper."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location(
    "coding_batch_study", "scripts/v3_coding/batched_study.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class BatchTests(unittest.TestCase):
    def test_client_preserves_frozen_prompt_and_token_cap(self):
        class Client:
            timings = [{"tokens_in": 2, "tokens_out": 3}]

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
