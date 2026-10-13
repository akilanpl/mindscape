"""One committed data qualification; stop at first failed cohort, no neural calls."""

import hashlib
import json
import math
import subprocess
from pathlib import Path

from mindscape.v3.schema import digest

from .data import graph
from .execute import execute


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def check(rows, domain, split, protocol):
    selected = [r for r in rows if r["domain"] == domain and r["split"] == split]
    selected = selected[:12] if split == "train" else selected
    valid = sum(r["target"]["valid"] for r in selected)
    changed = sum(
        r["target"]["valid"] and any(abs(v) > 1e-8 for a in r["target"]["delta"] for v in a)
        for r in selected
    )
    vals = [v * v for r in selected for a in r["target"]["delta"] for v in a]
    mse = sum(vals) / len(vals)
    t = protocol["qualification_gates"][
        "per_domain_first_12_training" if split == "train" else "per_domain_qualification"
    ]
    passed = (
        valid >= t["minimum_valid"]
        and len(selected) - valid >= t["minimum_invalid"]
        and changed >= t["minimum_changed_valid"]
    )
    if split == "qualification":
        passed = passed and mse >= t["minimum_persistence_normalized_mse"]
    return {
        "pass": passed,
        "tasks": len(selected),
        "valid": valid,
        "invalid": len(selected) - valid,
        "changed_valid": changed,
        "persistence_normalized_mse": mse,
    }


def qualify(manifest_path, root):
    raw = Path(manifest_path).read_bytes()
    if subprocess.check_output(["git", "show", "HEAD:" + str(manifest_path)]) != raw:
        raise ValueError("Complete task/protocol manifest must be committed before execution")
    manifest = json.loads(raw)
    protocol = manifest["protocol"]
    if (
        digest(manifest["tasks"]) != manifest["task_hash"]
        or digest(protocol) != manifest["protocol_hash"]
    ):
        raise ValueError("Freeze mismatch")
    root.mkdir(parents=True, exist_ok=False)
    rows = []
    sealed = []
    checks = {}
    failure = None
    stopped = False
    save(root / "manifest.json", manifest)
    try:
        for task in manifest["tasks"]:
            if task["split"] == "composition":
                continue
            g = graph(task)
            save(root / f"{task['identity']}-public.json", {"task": task, "graph": g})
            target = execute(task)
            row = {
                "identity": task["identity"],
                "domain": task["domain"],
                "split": task["split"],
                "graph": g,
                "target": target,
            }
            rows.append(row)
            save(root / "rows.json", rows)
            if (
                not all(math.isfinite(v) for a in target["delta"] for v in a)
                or execute(task) != target
            ):
                raise ValueError("Independent target replay/finite gate")
            count = sum(r["domain"] == task["domain"] and r["split"] == task["split"] for r in rows)
            if count == (12 if task["split"] == "train" else 8):
                value = check(rows, task["domain"], task["split"], protocol)
                checks[task["domain"] + "." + task["split"]] = value
                print(
                    json.dumps({"cohort": task["domain"] + "." + task["split"], **value}),
                    flush=True,
                )
                if not value["pass"]:
                    stopped = True
                    break
        if not stopped:
            for task in manifest["tasks"]:
                if task["split"] != "composition":
                    continue
                g = graph(task)
                save(root / f"{task['identity']}-public.json", {"task": task, "graph": g})
                target = execute(task)
                if execute(task) != target or not all(
                    math.isfinite(v) for a in target["delta"] for v in a
                ):
                    raise ValueError("Held-out target integrity replay")
                row = {
                    "identity": task["identity"],
                    "domain": task["domain"],
                    "split": task["split"],
                    "graph": g,
                    "target": target,
                }
                p = root / f"{task['identity']}-sealed.json"
                save(p, row)
                sealed.append(
                    {
                        "identity": task["identity"],
                        "file": p.name,
                        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                    }
                )
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        failure = str(exc)
    save(root / "rows.json", rows)
    save(root / "sealed-index.json", sealed)
    ready = (
        not failure
        and not stopped
        and len(checks) == 6
        and all(c["pass"] for c in checks.values())
        and len(sealed) == 36
    )
    summary = {
        "decision": "DATA_READY" if ready else "DATA_NO_GO",
        "gates": checks,
        "failure": failure,
        "independent_execution_replay": "PASS" if not failure else "FAILED_OR_INCOMPLETE",
        "executed_tasks": len(rows) + len(sealed),
        "frozen_tasks": len(manifest["tasks"]),
        "training_executed": False,
        "composition_targets_used_for_qualification": False,
    }
    save(root / "qualification.json", summary)
    print(json.dumps(summary), flush=True)
    return summary
