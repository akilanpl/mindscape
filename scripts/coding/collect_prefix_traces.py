"""Reexecute the accepted preflight prefix to preserve complete actual tuples."""

import json
from pathlib import Path

from mindscape.coding.policy import teacher_memory
from mindscape.coding.sandbox import WasiSandbox
from mindscape.coding.schema import CodeTask

root = Path("results/coding/final_retrieval_v1")
dataset = json.loads((root / "dataset.json").read_text())
records = json.loads((root / "teacher_memory.json").read_text())
sandbox = WasiSandbox("work/coding/runtime")
for i, (value, prior) in enumerate(zip(dataset["train"], records)):
    task = CodeTask(**value)
    if prior["initial_repository"] != task.repository:
        raise RuntimeError("Teacher prefix alignment failure")
    if (Path("results/coding/teacher_traces_v1") / (task.task_id + ".json")).exists():
        continue
    actual = teacher_memory([task], sandbox, 1).episodic[0]
    if actual != prior:
        raise RuntimeError("Reexecuted compact teacher record differs")
    if (i + 1) % 50 == 0:
        print("full actual prefix tuples", i + 1, flush=True)
print("Prefix tuple collection completed", len(records), flush=True)
