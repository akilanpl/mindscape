"""Frozen one-shot 100-task comparison; resume exact unfinished records only."""

import argparse
import gc
import json
from pathlib import Path

from mindscape.coding.instrumentation import StageMeter
from mindscape.coding.memory import CodingMemory
from mindscape.coding.model import LocalCoder
from mindscape.coding.policy import CodingPolicy, ToolCodingPolicy
from mindscape.coding.sandbox import WasiSandbox
from mindscape.coding.schema import CodeTask

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--root", default="results/coding/completion_lockbox_eval_v1")
a = p.parse_args()
root = Path(a.root)
root.mkdir(parents=True, exist_ok=True)
tasks = json.loads(Path("results/coding/completion_lockbox_v1/tasks.json").read_text())
records = json.loads(
    Path("results/coding/completion_collection_v1/teacher_memory.json").read_text()
)
memory = CodingMemory(episodic=records[:100])
sandbox = WasiSandbox("work/coding/runtime")
path = root / "rows.jsonl"
rows = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
done = {(r["condition"], r["task_id"]) for r in rows}
for condition in ("model_only", "structured", "mindscape_b", "mindscape_c"):
    adapter = (
        Path("results/coding/gradient_v1/seed_11/checkpoint_100")
        if condition == "model_only"
        else Path("results/coding/completion_gradient_v1") / condition / "seed_11/checkpoint_100"
    )
    if all((condition, t["task_id"]) in done for t in tasks):
        continue
    coder = LocalCoder(a.model, adapter=adapter)
    policy = (
        ToolCodingPolicy(coder, sandbox, memory)
        if condition.startswith("mindscape")
        else CodingPolicy(coder, sandbox, condition)
    )
    for value in tasks:
        if (condition, value["task_id"]) in done:
            continue
        with StageMeter() as meter:
            row = policy.solve(CodeTask(**value), 11)
        row.update(
            condition=condition,
            split=value["metadata"]["split"],
            seed=11,
            budget=100,
            stages=meter.summary(),
            one_shot_configuration=True,
            adapter=str(adapter),
        )
        with path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        rows.append(row)
        done.add((condition, value["task_id"]))
        print(condition, len(rows), row["success"], flush=True)
    del policy, coder
    gc.collect()
(root / "complete.json").write_text(
    json.dumps(
        {
            "complete": True,
            "tasks": 100,
            "conditions": 4,
            "episodes": len(rows),
            "selection": "preregistered budget100/seed11; no lockbox tuning",
        },
        indent=2,
    )
)
