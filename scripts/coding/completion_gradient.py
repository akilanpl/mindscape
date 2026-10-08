"""Resumable matched gradient comparison, honest separate retrieval budgets."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--seeds", default="11,23,37")
p.add_argument("--budgets", default="10,25,50,100")
p.add_argument("--root", default="results/coding/completion_gradient_v1")
p.add_argument("--train-only", action="store_true")
a = p.parse_args()
root = Path(a.root)
root.mkdir(parents=True, exist_ok=True)
seeds = [int(s) for s in a.seeds.split(",")]
budgets = [int(n) for n in a.budgets.split(",")]
# Serialize large resident models after the preexisting training pipeline.
while not Path("results/coding/gradient_v1/manifest.json").exists():
    time.sleep(5)
for condition in ("structured", "mindscape_b", "mindscape_c"):
    for seed in seeds:
        folder = root / condition / f"seed_{seed}"
        if (folder / "training.json").exists():
            continue
        subprocess.run(
            [
                sys.executable,
                "scripts/coding/train_regime.py",
                "--model",
                a.model,
                "--condition",
                condition,
                "--seed",
                str(seed),
                "--samples",
                str(max(budgets)),
                "--output",
                str(folder),
            ],
            check=True,
        )
if a.train_only:
    raise SystemExit(0)
# Import the heavy runtime only after training subprocesses release their models.
import gc

from mindscape.coding.instrumentation import StageMeter
from mindscape.coding.memory import CodingMemory
from mindscape.coding.model import LocalCoder
from mindscape.coding.policy import CodingPolicy, ToolCodingPolicy
from mindscape.coding.sandbox import WasiSandbox
from mindscape.coding.schema import CodeTask

locked = json.loads(Path("results/coding/final_dataset_v1/dataset.json").read_text())
teacher = json.loads(Path("results/coding/final_retrieval_v1/teacher_memory.json").read_text())
sandbox = WasiSandbox("work/coding/runtime")
rows_path = root / "rows.jsonl"
done = (
    {tuple(json.loads(s)["key"]) for s in rows_path.read_text().splitlines()}
    if rows_path.exists()
    else set()
)
for condition in ("model_only", "structured", "mindscape_b", "mindscape_c"):
    for seed in seeds:
        for budget in budgets:
            adapter = (
                Path("results/coding/gradient_v1") / f"seed_{seed}" / f"checkpoint_{budget}"
                if condition == "model_only"
                else root / condition / f"seed_{seed}" / f"checkpoint_{budget}"
            )
            keys = [
                (condition, seed, budget, split, v["task_id"])
                for split in ("test", "ood_test")
                for v in locked[split]
            ]
            if all(key in done for key in keys):
                continue
            coder = LocalCoder(a.model, adapter=adapter)
            memory = CodingMemory(episodic=teacher[:budget])
            policy = (
                ToolCodingPolicy(coder, sandbox, memory)
                if condition.startswith("mindscape")
                else CodingPolicy(coder, sandbox, condition)
            )
            for split in ("test", "ood_test"):
                for value in locked[split]:
                    key = (condition, seed, budget, split, value["task_id"])
                    if key in done:
                        continue
                    with StageMeter() as meter:
                        row = policy.solve(CodeTask(**value), seed)
                    row.update(
                        condition=condition,
                        key=key,
                        seed=seed,
                        budget=budget,
                        split=split,
                        stages=meter.summary(),
                        adapter=str(adapter),
                        gradient_examples=budget,
                        retrieval_examples=budget if condition.startswith("mindscape") else 0,
                    )
                    with rows_path.open("a") as f:
                        f.write(json.dumps(row) + "\n")
                    done.add(key)
                print(condition, seed, budget, split, "complete", flush=True)
            del policy, coder
            gc.collect()
(root / "complete.json").write_text(
    json.dumps(
        {
            "complete": True,
            "seeds": seeds,
            "budgets": budgets,
            "records": len(done),
            "supervision_caveat": "Action labels and actual past feedback differ across conditions; matched backbone and task counts alone do not isolate architecture.",
        },
        indent=2,
    )
)
