"""Independent execution evaluation of an actual LoRA checkpoint."""

import argparse
import json
from pathlib import Path

from mindscape.coding.generator import generate
from mindscape.coding.model import LocalCoder
from mindscape.coding.policy import CodingPolicy
from mindscape.coding.sandbox import WasiSandbox

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--adapter", required=True)
p.add_argument("--test", type=int, default=20)
p.add_argument("--output", default="results/coding/lora_probe_evaluation_v1")
a = p.parse_args()
root = Path(a.output)
root.mkdir(parents=True, exist_ok=True)
coder = LocalCoder(a.model, adapter=a.adapter)
sandbox = WasiSandbox("work/coding/runtime")
splits = generate(counts={"train": 0, "validation": 0, "test": a.test, "ood_test": a.test})
# Use the exact study's held-out tasks, not fresh random tasks with shifted RNG.
locked = json.loads(Path("results/coding/study_v1/dataset.json").read_text())
from mindscape.coding.schema import CodeTask

rows = []
for condition in ("model_only", "structured"):
    policy = CodingPolicy(coder, sandbox, condition)
    for split in ("test", "ood_test"):
        for value in locked[split][: a.test]:
            row = policy.solve(CodeTask(**value))
            row["split"] = split
            rows.append(row)
            with (root / "rows.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(condition, split, len(rows), row["success"], flush=True)
summary = {
    "model_revision": coder.revision,
    "cells": [
        {
            "condition": condition,
            "split": split,
            "tasks": len(group),
            "success_rate": sum(r["success"] for r in group) / len(group),
        }
        for condition in ("model_only", "structured")
        for split in ("test", "ood_test")
        if (group := [r for r in rows if r["condition"] == condition and r["split"] == split])
    ],
}
(root / "summary.json").write_text(json.dumps(summary, indent=2))
