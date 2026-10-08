"""Zero-domain-training controls: thresholds may be zero, DER then undefined."""

import argparse
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
p.add_argument("--output", default="results/coding/zero_budget_v1")
a = p.parse_args()
root = Path(a.output)
root.mkdir(parents=True, exist_ok=True)
locked = json.loads(Path("results/coding/final_dataset_v1/dataset.json").read_text())
coder = LocalCoder(a.model)
sandbox = WasiSandbox("work/coding/runtime")
path = root / "rows.jsonl"
done = (
    {(r["condition"], r["task_id"]) for r in [json.loads(s) for s in path.read_text().splitlines()]}
    if path.exists()
    else set()
)
for condition in ("model_only", "structured", "mindscape_b", "mindscape_c"):
    policy = (
        ToolCodingPolicy(coder, sandbox, CodingMemory())
        if condition.startswith("mindscape")
        else CodingPolicy(coder, sandbox, condition)
    )
    for split in ("test", "ood_test"):
        for value in locked[split]:
            if (condition, value["task_id"]) in done:
                continue
            with StageMeter() as meter:
                row = policy.solve(CodeTask(**value))
            row.update(
                condition=condition,
                split=split,
                budget=0,
                seed=0,
                gradient_examples=0,
                retrieval_examples=0,
                stages=meter.summary(),
                key=(condition, 0, 0, split, value["task_id"]),
            )
            with path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            done.add((condition, value["task_id"]))
            print(condition, split, len(done), row["success"], flush=True)
(root / "complete.json").write_text(
    json.dumps(
        {
            "episodes": len(done),
            "zero_training": True,
            "zero_memory": True,
            "repeated_frozen_controls_are_not_independent_seeds": True,
        },
        indent=2,
    )
)
