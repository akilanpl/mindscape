"""Actual lockbox oracle/defect integrity; no model inference or tuning."""

import hashlib
import json
from pathlib import Path

from mindscape.coding.sandbox import WasiSandbox
from mindscape.coding.testing import run_cases

root = Path("results/coding/completion_lockbox_validation_v1")
root.mkdir(parents=True, exist_ok=True)
source = Path("results/coding/completion_lockbox_v1/tasks.json")
tasks = json.loads(source.read_text())
path = root / "rows.jsonl"
done = (
    {tuple(json.loads(s)["key"]) for s in path.read_text().splitlines()} if path.exists() else set()
)
sandbox = WasiSandbox("work/coding/runtime")
for t in tasks:
    for version in ("repository", "correct_repository"):
        key = (t["task_id"], version)
        if key in done:
            continue
        result = run_cases(sandbox, t[version], t["hidden_tests"])
        if result["all_passed"] != (version == "correct_repository"):
            raise RuntimeError("Incorrect locked oracle/defect " + t["task_id"])
        with path.open("a") as f:
            f.write(json.dumps({"key": key, "result": result}) + "\n")
        done.add(key)
    print("validated lockbox pairs", len(done) // 2, flush=True)
(root / "complete.json").write_text(
    json.dumps(
        {
            "tasks": len(tasks),
            "executions": len(done),
            "dataset_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "actual_function_invocations": sum(
                len(json.loads(t["hidden_tests"])["cases"]) * 2 for t in tasks
            ),
            "all_passed": True,
        },
        indent=2,
    )
)
