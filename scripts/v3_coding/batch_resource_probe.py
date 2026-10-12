"""Training-only numerical/throughput probe; contains no held-out task."""

import argparse
import json
import time
from pathlib import Path

from batched_study import CpuBatchCoder

from mindscape.v3.coding_learning.data import load_public

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    tasks, _ = load_public("configs/v3_coding/benchmark_v1")
    tasks = [t for t in tasks if t.split == "train"][:2]
    coder = CpuBatchCoder(a.model, "work/coding-batch-probe-cache")
    requests = [
        ("Return the repaired Python source only.", json.dumps(t.visible()), 32, 11) for t in tasks
    ]
    start = time.perf_counter()
    scalar = [coder.generate(*r) for r in requests]
    scalar_time = time.perf_counter() - start
    start = time.perf_counter()
    batched = coder.generate_batch(requests)
    batch_time = time.perf_counter() - start
    Path(a.output).write_text(
        json.dumps(
            {
                "scope": "training-only throughput probe, not repair accuracy",
                "training_task_ids": [t.identity for t in tasks],
                "scalar_seconds": scalar_time,
                "batch_seconds": batch_time,
                "identical_responses": [s == b[0] for s, b in zip(scalar, batched, strict=True)],
                "scalar_responses": scalar,
                "batch_responses": batched,
                "model_loaded_seconds": coder.load_seconds,
            },
            indent=2,
        )
    )
