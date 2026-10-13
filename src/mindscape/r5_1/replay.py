"""Independent read-only result audit: no neural calls and no fresh/held-out tasks executed."""

import hashlib
import json
from pathlib import Path

from mindscape.v3.schema import digest
from mindscape.v5.data import graph
from mindscape.v5.execute import execute

from .metrics import baselines
from .qualification import cohort, save


def replay(root, output):
    root = Path(root)
    output = Path(output)
    if output.exists():
        raise FileExistsError("Audit output exists")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    preserved = json.loads(root.with_suffix(".preservation.json").read_text())
    before = {str(p.relative_to(root)): sha(p) for p in root.rglob("*") if p.is_file()}
    if (
        before != preserved["files"]
        or sha(root.with_suffix(".tar.gz")) != preserved["archive_sha256"]
    ):
        raise ValueError("Archive/run integrity")
    manifest = json.loads((root / "manifest.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    if (
        digest(manifest["tasks"]) != manifest["task_hash"]
        or digest(manifest["protocol"]) != manifest["protocol_hash"]
    ):
        raise ValueError("Frozen data/protocol hash")
    if summary["decision"] == "ENGINEERING_NO_GO":
        result = {
            "status": "PRESERVED_ENGINEERING_FAILURE",
            "decision": summary["decision"],
            "failure": summary["failure"],
            "neural_execution": False,
        }
    else:
        eng = json.loads((root / "engineering.json").read_text())
        if (
            eng["status"] != "PASS"
            or eng["optimizer_steps"] != 0
            or eng["parameters"] != 6967
            or not eng["weights_unchanged"]
        ):
            raise ValueError("Engineering receipt inconsistent")
        rows = json.loads((root / "rows.json").read_text())
        tasks = [t for t in manifest["tasks"] if t["split"] != "composition"]
        gates = {}
        counts = {}
        for i, row in enumerate(rows):
            task = tasks[i]
            if (
                row["task"] != task
                or row["identity"] != task["identity"]
                or row["graph"] != graph(task)
            ):
                raise ValueError("Task/graph binding")
            if (
                digest(execute(task)) != digest(row["target"])
                or baselines(task, row["target"]) != row["baselines"]
            ):
                raise ValueError("Independent effect/baseline mismatch")
            public = json.loads((root / "public" / f"{task['identity']}.json").read_text())
            if public != {"task": task, "graph": graph(task)}:
                raise ValueError("Public before-target receipt mismatch")
            key = (task["domain"], task["split"])
            counts[key] = counts.get(key, 0) + 1
            if counts[key] == manifest["protocol"]["tasks_per_domain"][task["split"]]:
                value = cohort(rows[: i + 1], *key, manifest["protocol"])
                gates[".".join(key)] = value
                if not value["pass"] and i + 1 != len(rows):
                    raise ValueError("Executed after failed gate")
        if gates != summary["gates"] or len(rows) != summary["executed_requests"] or len(rows) > 81:
            raise ValueError("Gate/budget summary mismatch")
        expected = (
            "DATA_READY"
            if len(gates) == 6
            and all(g["pass"] for g in gates.values())
            and len(rows) == 81
            and summary["failure"] is None
            else "DATA_NO_GO"
        )
        if expected != summary["decision"]:
            raise ValueError("Terminal decision mismatch")
        if any(row["split"] == "composition" for row in rows):
            raise ValueError("Held-out task executed")
        result = {
            "status": "PASS",
            "decision": expected,
            "executed_requests": len(rows),
            "gates": gates,
            "composition_targets_accessed": False,
            "neural_execution": False,
            "note": "Engineering values are recorded native receipts; this independent audit replays executed targets and gates, not neural forwards.",
        }
    if before != {str(p.relative_to(root)): sha(p) for p in root.rglob("*") if p.is_file()}:
        raise ValueError("Original evidence mutated")
    save(output, result)
    return result
