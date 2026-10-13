"""Read-only independent V5 qualification replay. Never imports neural model/training code."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from mindscape.v3.schema import digest
from mindscape.v5.data import graph
from mindscape.v5.execute import execute
from mindscape.v5.qualify import check


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def review(root, output):
    output.mkdir(parents=True, exist_ok=False)
    original = {str(p.relative_to(root)): sha(p) for p in root.rglob("*") if p.is_file()}
    manifest = json.loads((root / "manifest.json").read_text())
    protocol = manifest["protocol"]
    if (
        digest(protocol) != manifest["protocol_hash"]
        or digest(manifest["tasks"]) != manifest["task_hash"]
    ):
        raise ValueError("Manifest freeze changed")
    rows = json.loads((root / "rows.json").read_text())
    receipt = json.loads((root / "qualification.json").read_text())
    tasks = [t for t in manifest["tasks"] if t["split"] != "composition"]
    if len(rows) != 88 or receipt["decision"] != "DATA_NO_GO" or receipt["training_executed"]:
        raise ValueError("Original stop decision changed")
    {t["identity"]: t for t in tasks}
    seen = set()
    gates = {}
    counts = {}
    details = []
    for i, row in enumerate(rows):
        task = tasks[i]
        if (
            row["identity"] != task["identity"]
            or row["identity"] in seen
            or row["split"] != task["split"]
            or row["domain"] != task["domain"]
        ):
            raise ValueError("Task binding/order/overlap")
        seen.add(row["identity"])
        public = json.loads((root / f"{task['identity']}-public.json").read_text())
        if public != {"task": task, "graph": graph(task)} or row["graph"] != graph(task):
            raise ValueError("Public graph changed")
        if digest(execute(task)) != digest(row["target"]):
            raise ValueError("Independent executed effect mismatch")
        key = (row["domain"], row["split"])
        counts[key] = counts.get(key, 0) + 1
        cohort = 12 if row["split"] == "train" else 8
        if counts[key] == cohort:
            gate = check(rows[: i + 1], *key, protocol)
            gates[".".join(key)] = gate
            if not gate["pass"] and i + 1 != len(rows):
                raise ValueError("Execution continued after failed qualification gate")
        values = [v for a in row["target"]["delta"] for v in a]
        details.append(
            {
                "identity": row["identity"],
                "domain": row["domain"],
                "split": row["split"],
                "positive_valid": row["target"]["valid"],
                "negative_invalid": 1 - row["target"]["valid"],
                "changed": int(any(abs(v) > 1e-8 for v in values)),
                "scalar_effects": len(values),
                "persistence_normalized_mse": sum(v * v for v in values) / len(values),
            }
        )
    if gates != receipt["gates"] or not any(not v["pass"] for v in gates.values()):
        raise ValueError("Gate replay changed")
    if json.loads((root / "sealed-index.json").read_text()) or list(root.glob("*-sealed.json")):
        raise ValueError("Composition targets executed after data failure")
    if original != {str(p.relative_to(root)): sha(p) for p in root.rglob("*") if p.is_file()}:
        raise ValueError("Original receipts changed")
    with (output / "task-labels.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(details[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(details)
    result = {
        "status": "PASS",
        "decision": "DATA_NO_GO",
        "independent_tasks_replayed": len(rows),
        "gates": gates,
        "executed_counts": [
            {
                "domain": d,
                "split": s,
                "tasks": n,
                "positive_valid": sum(
                    r["target"]["valid"] for r in rows if r["domain"] == d and r["split"] == s
                ),
                "negative_invalid": sum(
                    1 - r["target"]["valid"] for r in rows if r["domain"] == d and r["split"] == s
                ),
            }
            for (d, s), n in counts.items()
        ],
        "unexecuted_tasks": len(manifest["tasks"]) - len(rows),
        "neural_execution": False,
        "original_files": original,
        "qualification_unchanged": True,
        "limits": [
            "Valid/invalid labels describe action preconditions, not candidate-repair correctness.",
            "Effects reproduced by real execution; no neural model was evaluated.",
        ],
    }
    (output / "independent-replay.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "PASS",
                "tasks": len(rows),
                "decision": result["decision"],
                "unexecuted": result["unexecuted_tasks"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    review(args.run, args.output)
