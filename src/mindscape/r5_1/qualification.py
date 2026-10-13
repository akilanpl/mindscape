"""One bounded independent execution screen after native engineering PASS."""

import json
from collections import Counter

from mindscape.v3.schema import digest
from mindscape.v5.data import graph
from mindscape.v5.execute import execute

from .metrics import aggregate, baselines


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def cohort(rows, domain, split, protocol):
    selected = [r for r in rows if r["domain"] == domain and r["split"] == split]
    persistence = aggregate([r["baselines"]["persistence"] for r in selected])
    local = aggregate([r["baselines"]["local_action"] for r in selected])
    t = protocol["qualification_gates_per_domain_and_split"][split]
    q = protocol["qualification_gates_per_domain_and_split"]["both_splits"]
    pp = persistence["changed_entity_mse"]
    ll = local["changed_entity_mse"]
    reduction = (pp - ll) / pp if pp and ll is not None else None
    counts = persistence
    checks = {
        "valid": counts["positive_valid"] >= t["minimum_valid"],
        "invalid": counts["negative_invalid"] >= t["minimum_invalid"],
        "changed": counts["tasks_with_changed_entities"] >= t["minimum_changed_valid"],
        "persistence_headroom": pp is not None
        and pp >= q["minimum_changed_entity_persistence_mse"],
        "local_headroom": ll is not None and ll >= q["minimum_changed_entity_local_action_mse"],
        "simple_learnable_signal": reduction is not None
        and reduction >= q["minimum_local_action_relative_reduction_vs_persistence"],
    }
    balance = None
    if domain == "coding":
        balance = Counter(
            (r["task"]["public"]["operations"][0], r["target"]["valid"]) for r in selected
        )
        expected = (4, 2) if split == "development" else (2, 1)
        checks["operator_validity_balance"] = all(
            balance[op, 1] == expected[0] and balance[op, 0] == expected[1]
            for op in ("add", "sub", "mul")
        )
        balance = {
            op: {"valid": balance[op, 1], "invalid": balance[op, 0]} for op in ("add", "sub", "mul")
        }
    # Retain V5-compatible scalar-pooled whole-graph metric as well as macro metrics.
    errors = [v * v for r in selected for a in r["target"]["delta"] for v in a]
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "persistence": persistence,
        "local_action": local,
        "local_action_relative_reduction": reduction,
        "operator_validity_balance": balance,
        "persistence_scalar_pooled_whole_graph_mse": sum(errors) / len(errors),
    }


def screen(manifest, root):
    protocol = manifest["protocol"]
    rows = []
    gates = {}
    failure = None
    (root / "public").mkdir()
    (root / "executed").mkdir()
    tasks = [t for t in manifest["tasks"] if t["split"] != "composition"]
    try:
        for task in tasks:
            if len(rows) >= protocol["maximum_qualification_executions"]:
                raise ValueError("Execution budget exceeded")
            g = graph(task)
            if digest(g) != manifest["public_graph_hashes"][task["identity"]]:
                raise ValueError("Frozen public graph mismatch")
            save(root / "public" / f"{task['identity']}.json", {"task": task, "graph": g})
            target = execute(task)
            row = {
                "identity": task["identity"],
                "domain": task["domain"],
                "split": task["split"],
                "task": task,
                "graph": g,
                "target": target,
                "baselines": baselines(task, target),
            }
            save(root / "executed" / f"{task['identity']}.json", row)
            if digest(execute(task)) != digest(target):
                raise ValueError("Independent exact outcome replay failed")
            rows.append(row)
            save(root / "rows.json", rows)
            n = sum(r["domain"] == task["domain"] and r["split"] == task["split"] for r in rows)
            if n == protocol["tasks_per_domain"][task["split"]]:
                value = cohort(rows, task["domain"], task["split"], protocol)
                gates[task["domain"] + "." + task["split"]] = value
                print(
                    json.dumps({"cohort": task["domain"] + "." + task["split"], **value}),
                    flush=True,
                )
                if not value["pass"]:
                    break
    except (RuntimeError, ValueError, OSError, KeyError, TypeError) as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
    ready = (
        failure is None
        and len(gates) == 6
        and all(g["pass"] for g in gates.values())
        and len(rows) == 81
    )
    summary = {
        "decision": "DATA_READY" if ready else "DATA_NO_GO",
        "gates": gates,
        "failure": failure,
        "executed_requests": len(rows),
        "unexecuted_screen_requests": 81 - len(rows),
        "unexecuted_composition_tasks": 27,
        "replayed_outcomes": len(rows),
        "neural_training": False,
        "checkpoint_updates": False,
        "next_action": "Request authorization for a bounded learning pilot"
        if ready
        else "STOP; no retries or training",
    }
    save(root / "summary.json", summary)
    return summary
