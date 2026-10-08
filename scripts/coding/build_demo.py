"""Build a private-data-free playback of actual frozen benchmark trajectories."""

import json
from pathlib import Path

source = Path("results/coding/completion_lockbox_eval_v1")
if not (source / "complete.json").exists():
    raise RuntimeError("Only completed locked evidence may populate demo")
tasks = {
    t["task_id"]: t
    for t in json.loads(Path("results/coding/completion_lockbox_v1/tasks.json").read_text())
}
rows = [json.loads(s) for s in (source / "rows.jsonl").read_text().splitlines()]
records = []
for row in rows:
    if row["condition"] not in ("model_only", "mindscape_c"):
        continue
    task = tasks[row["task_id"]]
    records.append(
        {
            "condition": row["condition"],
            "task_id": row["task_id"],
            "problem": task["problem_statement"],
            "before": task["repository"],
            "after": row["repository"],
            "success": row["success"],
            "wall_seconds": row["wall_seconds"],
            "model_calls": row["model_calls"],
            "steps": [
                {
                    "action": t["action"],
                    "event": t["event"]["name"],
                    "valid": t["valid"],
                    "stdout": t["result"]["stdout"],
                    "stderr": t["result"]["stderr"],
                    "files": t["state_after"]["files"],
                    "symbols": t["state_after"]["symbols"],
                    "relations": t["state_after"]["relations"],
                }
                for t in row["trajectory"]["transitions"]
            ],
        }
    )
root = Path("demo/coding")
root.mkdir(parents=True, exist_ok=True)
(root / "evidence.json").write_text(json.dumps(records))
template = (root / "template.html").read_text()
(root / "index.html").write_text(
    template.replace("/*EVIDENCE*/[]", json.dumps(records).replace("</", "<\\/"))
)
print("Actual recorded episodes", len(records))
