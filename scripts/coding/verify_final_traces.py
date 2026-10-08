"""Independently replay every locked coding trajectory before publication."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from mindscape.coding.backend import CodingBackend
from mindscape.coding.environment import CodeRepairEnvironment
from mindscape.coding.schema import CodeAction, CodeTask
from mindscape.data.schemas import BenchmarkExample


def without_duration(value):
    if isinstance(value, dict):
        return {k: without_duration(v) for k, v in value.items() if k != "duration"}
    if isinstance(value, (tuple, list)):
        return [without_duration(v) for v in value]
    return value


def replay_evidence(task, trajectory, answer, sandbox, detailed=False):
    env = CodeRepairEnvironment(sandbox)
    env.reset(CodeTask(**task))
    matches = without_duration(asdict(env.get_state())) == without_duration(
        trajectory["initial_state"]
    )
    mismatches = []
    state_checks = [matches]
    tool_checks = []
    try:
        for index, recorded in enumerate(trajectory["transitions"]):
            actual = asdict(env.step(CodeAction(**recorded["action"])))
            state_checks.extend(without_duration(actual[k]) == without_duration(recorded[k])
                                for k in ("state_before", "state_after"))
            tool_checks.append(all(without_duration(actual[k]) == without_duration(recorded[k])
                                   for k in ("action", "event", "result", "valid")))
            if without_duration(actual) != without_duration(recorded):
                matches = False
                mismatches.append(index)
        matches = matches and env.files == answer
    finally:
        env.close()
    detail = {"state_agreement": sum(state_checks)/len(state_checks),
              "state_comparisons": len(state_checks), "state_matches": sum(state_checks),
              "tool_result_agreement": sum(tool_checks)/len(tool_checks) if tool_checks else None,
              "tool_result_comparisons": len(tool_checks), "tool_result_matches": sum(tool_checks)}
    return (matches, mismatches, detail) if detailed else (matches, mismatches)


parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--partial",action="store_true",help="Audit all actual saved rows while retaining incomplete lockbox scope")
args=parser.parse_args()

root = Path("results/coding/completion_trace_audit_v1")
root.mkdir(parents=True, exist_ok=True)
rows_path = Path("results/coding/completion_lockbox_eval_v1/rows.jsonl")
if not args.partial and not any(Path("results/coding/completion_lockbox_eval_v1",marker).exists()
           for marker in ("complete.json","primary_complete.json")):
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
    evidence_matches, mismatches, detail = replay_evidence(
        task, row["trajectory"], row["repository"], backend._sandbox(), detailed=True
    )
    record = {
        **detail,
        "key": key,
        "trajectory_valid": valid,
        "goal": goal,
        "matches_reported_goal": goal == row["success"],
        "error": error,
        "structured_evidence_matches": evidence_matches,
        "mismatching_transition_indices": mismatches,
        "comparison_scope": "All typed transition fields and states; measured durations excluded",
        "replay_passes": 2,
    }
    if goal != row["success"]:
        raise RuntimeError("Independent terminal disagreement")
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")
    done.add(key)
    print("independent replay", len(done), flush=True)
saved = {(r["condition"],r["task_id"]) for r in map(json.loads,rows_path.read_text().splitlines())}
if done != saved:
    raise RuntimeError("Saved locked rows changed during audit; rerun incrementally before publication")
(root / "complete.json").write_text(json.dumps({"episodes":len(done),"complete":True,
    "scope":"Independent audit of all actual saved locked episodes, not unrun rows",
    "original_four_condition_study_episodes":400,"four_condition_study_complete":len(done)==400}))
