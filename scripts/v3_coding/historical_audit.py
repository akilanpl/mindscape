"""Read historical result evidence only; never open or rerun historical lockbox tasks."""

import argparse
import json
from collections import Counter
from pathlib import Path


def audit(root, output):
    rows = []
    for p in sorted(Path(root).glob("lockbox_*.jsonl")):
        rows.extend(json.loads(line) for line in p.read_text().splitlines() if line.strip())
    if not rows:
        p = Path(root) / "rows.jsonl"
        rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    result = {}
    paired = {}
    for condition in ("mindscape_b", "mindscape_c"):
        cases = [r for r in rows if r.get("condition") == condition]
        counts = Counter()
        actions = Counter()
        errors = Counter()
        plans = Counter()
        for r in cases:
            counts["episodes"] += 1
            counts["success"] += bool(r.get("success"))
            counts["model_calls"] += r.get("model_calls", 0)
            success_at = None
            for index, a in enumerate(r.get("attempts", [])):
                if a.get("parse_error"):
                    errors[a["parse_error"]] += 1
                action = a.get("action") or {}
                actions[action.get("name", "unparsed")] += 1
                if a.get("visible_pass") is True and success_at is None:
                    success_at = index
                for branch in a.get("planning", []):
                    plans[str(branch.get("score"))] += 1
            if success_at is not None:
                counts["episodes_with_visible_pass"] += 1
                counts["actions_after_first_visible_pass"] += (
                    len(r.get("attempts", [])) - success_at - 1
                )
            if not r.get("success") and len(r.get("attempts", [])) >= 8:
                counts["failure_at_action_cap"] += 1
            paired.setdefault(r["task_id"], {})[condition] = bool(r.get("success"))
        result[condition] = {
            "by_split": {
                split: {
                    "episodes": sum(r.get("split") == split for r in cases),
                    "successes": sum(
                        bool(r.get("success")) for r in cases if r.get("split") == split
                    ),
                }
                for split in ("test", "ood_test")
            },
            "counts": dict(counts),
            "actions": dict(actions),
            "errors": dict(errors),
            "planner_score_histogram": dict(plans),
        }
    result["paired_all_saved_tasks"] = {
        "b_only": sum(
            v.get("mindscape_b", False) and not v.get("mindscape_c", False) for v in paired.values()
        ),
        "c_only": sum(
            v.get("mindscape_c", False) and not v.get("mindscape_b", False) for v in paired.values()
        ),
        "both": sum(
            v.get("mindscape_c", False) and v.get("mindscape_b", False) for v in paired.values()
        ),
    }
    result["limits"] = (
        "Saved historical episodes; mixed hardware, adapters and supervision. Descriptive evidence, not a causal mechanism test. Historical task files and private expectations never opened."
    )
    Path(output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    audit(a.root, a.output)
