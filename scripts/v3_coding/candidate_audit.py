"""Post-termination counterfactual proposal execution; never fed back to learner."""

import argparse
import json
from pathlib import Path

from mindscape.v3.coding_learning.data import load_public, load_scorer
from mindscape.v3.coding_learning.execution import Executor
from mindscape.v3.coding_learning.schema import RepairProposal
from mindscape.v3.schema import digest


def audit(args):
    tasks, _ = load_public(args.manifest)
    tasks = {t.identity: t for t in tasks}
    private = load_scorer(args.manifest)
    executor = Executor(args.runtime)
    rows, cache = [], {}
    for path in sorted(Path(args.run).glob("*-mindscape.json")):
        saved = json.loads(path.read_text())
        trace = saved["trace"]
        if digest(trace) != saved["trace_hash"]:
            raise ValueError("Trace changed")
        task = tasks[trace["task_id"]]
        for attempt in trace["attempts"]:
            if attempt["selected"] is None:
                continue
            outcomes = []
            for proposal in attempt["candidates"]:
                repository = RepairProposal(**proposal).apply(attempt["payload"]["repository"])
                key = digest([task.identity, repository])
                if key not in cache:
                    cache[key] = {
                        "visible": executor.run(repository, task.visible_tests),
                        "private": executor.run(
                            repository, private[task.identity]["private_tests"]
                        ),
                    }
                outcomes.append(cache[key])
            selected = attempt["selected"]
            rows.append(
                {
                    "task_id": task.identity,
                    "split": task.split,
                    "family": task.family,
                    "turn": attempt["turn"],
                    "selected": selected,
                    "predictions": attempt["predictions"],
                    "candidate_outcomes": outcomes,
                }
            )
    totals = {}
    for split in ("iid", "ood"):
        eligible = [r for r in rows if r["split"] == split and len(r["candidate_outcomes"]) == 2]
        totals[split] = {
            "two_candidate_decisions": len(eligible),
            "selected_nonzero": sum(r["selected"] != 0 for r in eligible),
            "selected_private_passes": sum(
                r["candidate_outcomes"][r["selected"]]["private"]["all_passed"] for r in eligible
            ),
            "candidate0_private_passes": sum(
                r["candidate_outcomes"][0]["private"]["all_passed"] for r in eligible
            ),
            "some_candidate_private_passes": sum(
                any(o["private"]["all_passed"] for o in r["candidate_outcomes"]) for r in eligible
            ),
            "private_outcomes_differ": sum(
                r["candidate_outcomes"][0]["private"]["all_passed"]
                != r["candidate_outcomes"][1]["private"]["all_passed"]
                for r in eligible
            ),
        }
    result = {
        "scope": "Post-termination candidate counterfactuals, not learner feedback or primary rollout accuracy",
        "episodes": len(list(Path(args.run).glob("*-mindscape.json"))),
        "rows": rows,
        "totals": totals,
        "actual_vm_requests": executor.calls,
        "model_calls": 0,
    }
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="configs/v3_coding/benchmark_v2")
    p.add_argument("--run", required=True)
    p.add_argument("--runtime", required=True)
    p.add_argument("--output", required=True)
    audit(p.parse_args())
