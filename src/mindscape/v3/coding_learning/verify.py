"""Replay numerical outcomes in the preserved WASI sandbox without model inference."""

import argparse
import json
from pathlib import Path

from mindscape.v3.schema import digest

from .data import load_public, load_scorer
from .execution import Executor
from .schema import RepairProposal


def replay(root, manifest, runtime):
    tasks, _ = load_public(manifest)
    tasks = {t.identity: t for t in tasks}
    private = load_scorer(manifest)
    executor = Executor(runtime)
    count = 0
    for path in Path(root).glob("*.json"):
        saved = json.loads(path.read_text())
        if "trace" not in saved:
            continue
        trace = saved["trace"]
        if saved["trace_hash"] != digest(trace):
            raise ValueError("Trace changed")
        task = tasks[trace["task_id"]]
        repo = dict(task.repository)
        snapshots = []
        if task.split not in ("iid", "ood"):
            raise ValueError("Evaluation partition changed")
        for a in trace["attempts"]:
            if a["selected"] is None:
                continue
            proposal = RepairProposal(**a["candidates"][a["selected"]])
            repo = proposal.apply(repo)
            result = executor.run(repo, task.visible_tests)
            if (
                result["all_passed"] != a["actual_visible_result"]["all_passed"]
                or result["case_passes"] != a["actual_visible_result"]["case_passes"]
            ):
                raise ValueError("Visible replay mismatch")
            snapshots.append(dict(repo))
        if repo != trace["repository"] or snapshots != trace["snapshots"]:
            raise ValueError("Transition changed")
        result = executor.run(repo, private[task.identity]["private_tests"])
        if result["all_passed"] != trace["success"]:
            raise ValueError("Terminal replay mismatch")
        if snapshots:
            first = executor.run(snapshots[0], private[task.identity]["private_tests"])
            if first["all_passed"] != trace["first_edit_success"]:
                raise ValueError("First-edit replay mismatch")
        count += 1
    return {"replayed_episodes": count, "sandbox_requests": executor.calls, "model_calls": 0}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--runtime", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    r = replay(a.run, a.manifest, a.runtime)
    Path(a.output).write_text(json.dumps(r, indent=2))
    print(json.dumps(r))
