"""Post hoc descriptive coverage and rollout diagnostics; no model selection."""

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

from mindscape.v3.math_learning.environment import LocalState, Problem, expected


def analyze(root, output):
    root = Path(root)
    known = set()
    with gzip.open(root / "demonstrations-1000.jsonl.gz", "rt") as stream:
        for line in stream:
            for e in json.loads(line)["events"]:
                if "state" in e:
                    known.add(LocalState(**e["state"]).key)
    records = []
    for directory in sorted(root.glob("eval-*-n1000-s*")):
        identity = json.loads((directory / "receipt.json").read_text())["identity"]
        with gzip.open(directory / "contexts.json.gz", "rt") as stream:
            contexts = json.load(stream)
        counts = Counter()
        by_context = defaultdict(Counter)
        first_errors = Counter()
        families = defaultdict(Counter)
        with gzip.open(directory / "traces.jsonl.gz", "rt") as stream:
            for line in stream:
                trace = json.loads(line)
                if trace["problem"]["split"] != "ood":
                    continue
                counts["tasks"] += 1
                counts["success"] += trace["success"]
                counts["strict"] += bool(trace["strict_rollout"])
                family = trace["problem"]["family"]
                families[family]["tasks"] += 1
                families[family]["success"] += trace["success"]
                task_local_error = False
                task_sign_error = False
                current = trace["problem"]["operands"][0]
                position = 0
                for event in trace["events"]:
                    if "state" in event:
                        state = LocalState(**event["state"])
                        target = event["posthoc_expected"]
                        claim = event["claim"]
                        next_ok = (claim["next_digit"], claim["next_carry"]) == (
                            target["digit"],
                            target["carry"],
                        )
                        group = (
                            "seen_train_context" if state.key in known else "unseen_train_context"
                        )
                        by_context[group]["steps"] += 1
                        by_context[group]["next_correct"] += next_ok
                        counts["steps"] += 1
                        counts["next_correct"] += next_ok
                        counts["incoming_nonzero_steps"] += state.incoming > 0
                        counts["incoming_nonzero_errors"] += state.incoming > 0 and not next_ok
                        counts["borrow_steps"] += state.operation == "sub"
                        counts["borrow_errors"] += (
                            state.operation == "sub" and claim["next_carry"] != target["carry"]
                        )
                        diag = contexts[event["context_key"]]["diagnostics"]
                        top2 = sorted(range(10), key=lambda i: diag["logits"][4][i], reverse=True)[
                            :2
                        ]
                        carrytop2 = sorted(
                            range(10), key=lambda i: diag["logits"][5][i], reverse=True
                        )[:2]
                        counts["true_next_outside_search"] += (
                            target["digit"] not in top2 or target["carry"] not in carrytop2
                        )
                        if not next_ok and not task_local_error:
                            first_errors[str(position)] += 1
                        task_local_error |= not next_ok
                        position += 1
                    else:
                        index = event["stage_result"]
                        p = Problem(
                            (current, trace["problem"]["operands"][index + 1]),
                            (trace["problem"]["operations"][index],),
                            "unit",
                            "diagnostic",
                        )
                        sign_ok = event["sign"] == int(expected(p) < 0)
                        counts["sign_decisions"] += 1
                        counts["sign_errors"] += not sign_ok
                        task_sign_error |= not sign_ok
                        current = int(event["value"])
                counts["tasks_with_next_error"] += task_local_error
                counts["tasks_with_sign_error"] += task_sign_error
                counts["failed_with_next_error"] += not trace["success"] and task_local_error
                counts["failed_with_sign_error"] += not trace["success"] and task_sign_error
        records.append(
            {
                "condition": identity["condition"],
                "seed": identity["seed"],
                "counts": dict(counts),
                "context_coverage": {k: dict(v) for k, v in by_context.items()},
                "first_next_error_position": dict(first_errors),
                "family_counts": {k: dict(v) for k, v in families.items()},
            }
        )
    result = {
        "budget": 1000,
        "unique_demonstrated_contexts": len(known),
        "records": records,
        "interpretation": "Post hoc descriptive diagnostics conditional on self-generated states; no tuning.",
    }
    Path(output).write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    analyze(a.run, a.output)
