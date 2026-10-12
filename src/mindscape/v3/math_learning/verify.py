"""Independent byte/row consistency plus scripted rollout replay without neural calls."""

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

from .environment import Claim, LocalState, Problem, run_problem
from .training import file_hash


def verify(root):
    counts = Counter()
    for path in sorted(Path(root).glob("eval-*/receipt.json")) + sorted(
        Path(root).glob("control-*/receipt.json")
    ):
        receipt = json.loads(path.read_text())
        for name, sha in receipt["files"].items():
            if file_hash(path.parent / name) != sha:
                raise ValueError("Evidence checksum mismatch")
        rows = {r["task_id"]: r for r in receipt["rows"]}
        if len(rows) != len(receipt["rows"]):
            raise ValueError("Duplicate task key")
        seen = set()
        with gzip.open(path.parent / "traces.jsonl.gz", "rt") as stream:
            for line in stream:
                trace = json.loads(line)
                p = trace["problem"]
                problem = Problem(
                    tuple(p["operands"]), tuple(p["operations"]), p["split"], p["family"]
                )
                if problem.identity != trace["task_id"] or problem.identity in seen:
                    raise ValueError("Task binding mismatch")
                seen.add(problem.identity)
                if trace["events"]:
                    locals_ = [e for e in trace["events"] if "state" in e]
                    signs = [e["sign"] for e in trace["events"] if "stage_result" in e]

                    class Replay:
                        def __init__(self, local_events, sign_values):
                            self.locals = local_events
                            self.signs = sign_values
                            self.local_index = 0
                            self.sign_index = 0

                        def predict_local(self, state):
                            event = self.locals[self.local_index]
                            self.local_index += 1
                            if state != LocalState(**event["state"]):
                                raise ValueError("State continuity mismatch")
                            return Claim(**event["claim"]), event["diagnostics"]

                        def predict_sign(self, visible):
                            sign = self.signs[self.sign_index]
                            self.sign_index += 1
                            return sign, {}

                    replayed = run_problem(problem, Replay(locals_, signs))
                    for key in (
                        "answer",
                        "expected",
                        "success",
                        "strict_rollout",
                        "steps",
                        "errors",
                    ):
                        if replayed[key] != trace[key]:
                            raise ValueError("Independent replay mismatch")
                else:
                    from .environment import expected

                    if trace["success"] != (int(trace["answer"]) == expected(problem)):
                        raise ValueError("Answer judgment mismatch")
                for key in ("success", "strict_rollout", "steps", "errors"):
                    if rows[problem.identity][key] != trace[key]:
                        raise ValueError("Summary mismatch")
                counts["executions"] += 1
                counts["successes"] += trace["success"]
        if seen != set(rows):
            raise ValueError("Incomplete execution set")
        counts["evaluation_groups"] += 1
    return dict(counts, neural_forward_calls=0)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    result = verify(a.run)
    Path(a.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
