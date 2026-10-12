"""Exploratory old-adapter bridge on two new public tasks; never old lockbox access."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from mindscape.coding.memory import CodingMemory
from mindscape.coding.model import LocalCoder
from mindscape.coding.policy import ToolCodingPolicy
from mindscape.coding.schema import CodeTask
from mindscape.v3.coding_learning.data import load_public, load_scorer
from mindscape.v3.coding_learning.execution import Executor
from mindscape.v3.coding_learning.memory import VerifiedMemory
from mindscape.v3.schema import digest

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--condition", choices=("mindscape_b", "mindscape_c"), required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--training", required=True)
    p.add_argument("--runtime", required=True)
    a = p.parse_args()
    root = Path(a.output)
    root.mkdir(parents=True, exist_ok=True)
    tasks, _ = load_public("configs/v3_coding/benchmark_v1")
    private = load_scorer("configs/v3_coding/benchmark_v1")
    selected = [
        next(t for t in tasks if t.split == "ood" and t.family == family)
        for family in ("cross_file", "compound")
    ]
    executor = Executor(a.runtime)
    verified = VerifiedMemory.load(
        Path(a.training) / "memory.json",
        [t for t in tasks if t.split == "train"],
        executor,
        private,
    )
    adapter = Path(a.adapter) / "adapter_model.safetensors"
    before = hashlib.sha256(adapter.read_bytes()).hexdigest()
    coder = LocalCoder(
        a.model, adapter=a.adapter, cache_enabled=False, device="cpu", precision="float32"
    )
    for t in selected:
        path = root / (t.identity + ".json")
        if path.exists():
            continue
        memory = CodingMemory(
            episodic=[
                {
                    "problem": r["contract"]["description"],
                    "initial_repository": r["before"],
                    "patch": r["after"],
                    "verified": True,
                    "hypothetical": False,
                }
                for r in verified.records
            ]
        )
        public = CodeTask(
            t.identity,
            t.repository,
            t.contract.description,
            json.dumps(t.visible_tests),
            None,
            {},
            {},
            {"split": t.split},
        )
        start = time.perf_counter()
        result = ToolCodingPolicy(coder, executor.sandbox, memory, max_steps=8, max_edits=3).solve(
            public, 11
        )
        terminal = executor.run(result["repository"], private[t.identity]["private_tests"])
        row = {
            "scope": "post hoc two-task historical-adapter bridge, not accuracy comparison to old48%",
            "task_id": t.identity,
            "family": t.family,
            "condition": a.condition,
            "old_adapter_sha256": before,
            "new_training_examples": 24,
            "legacy_budget": {"model_calls": 8, "edits": 3},
            "limits": "Different training history and action budget; mutation invariant feedback differs. No fair superiority inference.",
            "trace": result,
            "success": terminal["all_passed"],
            "independent_terminal": terminal,
            "seconds": time.perf_counter() - start,
            "trace_hash": digest(result),
        }
        path.write_text(json.dumps(row, indent=2))
        print(
            json.dumps({"condition": a.condition, "family": t.family, "success": row["success"]}),
            flush=True,
        )
    if hashlib.sha256(adapter.read_bytes()).hexdigest() != before:
        raise ValueError("Historical adapter changed")
