"""Independently replay every locked coding trajectory before publication."""

import json
from pathlib import Path

from mindscape.coding.backend import CodingBackend
from mindscape.data.schemas import BenchmarkExample

root = Path("results/coding/completion_trace_audit_v1")
root.mkdir(parents=True, exist_ok=True)
rows_path = Path("results/coding/completion_lockbox_eval_v1/rows.jsonl")
if not Path("results/coding/completion_lockbox_eval_v1/complete.json").exists():
    raise RuntimeError("Lockbox incomplete")
tasks = {
    t["task_id"]: t
    for t in json.loads(Path("results/coding/completion_lockbox_v1/tasks.json").read_text())
}
path = root / "rows.jsonl"
done = (
    {tuple(json.loads(s)["key"]) for s in path.read_text().splitlines()} if path.exists() else set()
)
backend = CodingBackend()
for line in rows_path.read_text().splitlines():
    row = json.loads(line)
    key = (row["condition"], row["task_id"])
    if key in done:
        continue
    task = tasks[row["task_id"]]
    example = BenchmarkExample(
        task["task_id"],
        "python_code_repair",
        task["problem_statement"],
        {"repository": task["repository"], "visible_tests": task["visible_tests"]},
        {},
        {},
        task["correct_repository"],
        [],
        {"private_tests": task["hidden_tests"]},
    )
    valid, goal, error = backend.assess(example, row["repository"], row["trajectory"])
    record = {
        "key": key,
        "trajectory_valid": valid,
        "goal": goal,
        "matches_reported_goal": goal == row["success"],
        "error": error,
    }
    if goal != row["success"]:
        raise RuntimeError("Independent terminal disagreement")
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")
    done.add(key)
    print("independent replay", len(done), flush=True)
(root / "complete.json").write_text(json.dumps({"episodes": len(done), "complete": True}))
