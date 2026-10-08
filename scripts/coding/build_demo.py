"""Build a private-data-free playback of actual frozen benchmark trajectories."""

import argparse
import json
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--partial",action="store_true",help="Publish explicitly partial saved evidence; never claim full C completion")
args=parser.parse_args()

source = Path("results/coding/completion_lockbox_eval_v1")
if not args.partial and not any((source / marker).exists() for marker in ("complete.json","primary_complete.json")):
    raise RuntimeError("Only completed locked evidence may populate demo")
tasks = {
    t["task_id"]: t
    for t in json.loads(Path("results/coding/completion_lockbox_v1/tasks.json").read_text())
}
rows = [json.loads(s) for s in (source / "rows.jsonl").read_text().splitlines()]
candidate = "mindscape_c" if any(r["condition"]=="mindscape_c" for r in rows) else "mindscape_b"
if candidate != "mindscape_c" and not args.partial:
    raise RuntimeError("Flagship C evidence missing")
shared_ids = {r["task_id"] for r in rows if r["condition"]=="model_only"} & {r["task_id"] for r in rows if r["condition"]==candidate}
records = []
for row in rows:
    if row["condition"] not in ("model_only", candidate) or row["task_id"] not in shared_ids:
        continue
    task = tasks[row["task_id"]]
    records.append(
        {
            "condition": row["condition"],
            "task_id": row["task_id"],
            "problem": task["problem_statement"],
            "before": task["repository"],
            "goal": row["trajectory"]["initial_state"]["goal"]["description"],
            "initial_state": {
                field: row["trajectory"]["initial_state"][field]
                for field in ("symbols", "relations", "current_changes", "failing_tests", "passing_tests", "errors", "progress")
            },
            "after": row["repository"],
            "success": row["success"],
            "terminal_passed": row["terminal"]["passed"],
            "terminal_total": row["terminal"]["total"],
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
                    "state": {
                        field: t["state_after"][field]
                        for field in ("current_changes", "failing_tests", "passing_tests", "errors", "progress")
                    },
                    "observation": t["state_after"]["observations"][-1],
                    "evidence_kind": t["result"]["evidence_kind"],
                }
                for t in row["trajectory"]["transitions"]
            ],
        }
    )
root = Path("demo/coding")
root.mkdir(parents=True, exist_ok=True)
(root / "evidence.json").write_text(json.dumps(records))
template = (root / "template.html").read_text()
if not records:
    raise RuntimeError("No paired actual evidence to populate")
if args.partial:
    template=template.replace("Replay the frozen 100-task study.",f"Replay actual saved evidence for {len(shared_ids)} of 100 locked tasks. Full study completion is not implied.")
    template=template.replace("Mindscape replay",f"{candidate} recorded replay")
    template=template.replace("mindscape_c",candidate)
(root / "index.html").write_text(
    template.replace("/*EVIDENCE*/[]", json.dumps(records).replace("</", "<\\/"))
)
print("Actual recorded episodes", len(records))
