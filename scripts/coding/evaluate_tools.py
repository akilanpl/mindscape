"""LoRA-adapted trajectory and experiential policies on the original locked tasks."""

import argparse
import json
from pathlib import Path

from mindscape.coding.memory import CodingMemory
from mindscape.coding.model import LocalCoder
from mindscape.coding.policy import CodingPolicy
from mindscape.coding.sandbox import WasiSandbox
from mindscape.coding.schema import CodeTask

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--adapter", required=True)
p.add_argument("--budget", type=int, default=10)
p.add_argument("--output", default="results/coding/lora_tools_v1")
a = p.parse_args()
root = Path(a.output)
root.mkdir(parents=True, exist_ok=True)
coder = LocalCoder(a.model, adapter=a.adapter)
sandbox = WasiSandbox("work/coding/runtime")
locked = json.loads(Path("results/coding/final_dataset_v1/dataset.json").read_text())
memory = CodingMemory(
    episodic=json.loads(Path("results/coding/study_v1/teacher_memory.json").read_text())[: a.budget]
)
rows_path = root / "rows.jsonl"
rows = [json.loads(s) for s in rows_path.read_text().splitlines()] if rows_path.exists() else []
done = {(r["condition"], r["split"], r["task_id"]) for r in rows}
for condition in ("trajectory", "experiential"):
    policy = CodingPolicy(coder, sandbox, condition, memory)
    for split in ("test", "ood_test"):
        for value in locked[split]:
            if (condition, split, value["task_id"]) in done:
                continue
            row = policy.solve(CodeTask(**value))
            row["split"] = split
            rows.append(row)
            with rows_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(condition, split, len(rows), row["success"], flush=True)
summary = {
    "model_revision": coder.revision,
    "gradient_samples": a.budget,
    "retrieval_samples": a.budget,
    "cells": [
        {
            "condition": condition,
            "split": split,
            "tasks": len(group),
            "success_rate": sum(r["success"] for r in group) / len(group),
        }
        for condition in ("trajectory", "experiential")
        for split in ("test", "ood_test")
        if (group := [r for r in rows if r["condition"] == condition and r["split"] == split])
    ],
}
(root / "summary.json").write_text(json.dumps(summary, indent=2))
