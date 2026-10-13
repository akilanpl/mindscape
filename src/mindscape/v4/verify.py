"""Nonneural arithmetic/WASI/Pymunk outcome replay, independent of learned predictions."""

import json
from pathlib import Path


def outcomes(task, proposals, executor=None):
    if task["domain"] == "mathematics":
        p = task["public"]
        raw = (
            p["left"] + p["right"] + p["incoming"]
            if p["operation"] == "add"
            else p["left"] - p["right"] - p["incoming"]
        )
        digit, carry = raw % 10, (raw // 10 if p["operation"] == "add" else int(raw < 0))
        from mindscape.v3.math_learning.environment import Claim, LocalState, execute_local

        results = []
        for c in proposals:
            written = execute_local(LocalState(**p), Claim(**c["claim"]))
            results.append(
                {
                    "success": written["written_digit"] == digit and written["incoming"] == carry,
                    "written_digit": written["written_digit"],
                    "written_carry": written["incoming"],
                }
            )
        return results
    if task["domain"] == "coding":
        if executor is None:
            from mindscape.v3.coding_learning.execution import Executor

            executor = Executor(Path("work/coding/runtime"))
        results = []
        for p in proposals:
            visible = executor.run(p["repository"], task["public"]["visible_tests"])
            private = executor.run(p["repository"], task["private"])
            results.append(
                {
                    "success": bool(visible["all_passed"] and private["all_passed"]),
                    "visible_pass": bool(visible["all_passed"]),
                    "private_pass": bool(private["all_passed"]),
                    "visible_observed": visible.get("observed", []),
                    "private_observed": private.get("observed", []),
                    "visible_case_passes": visible["case_passes"],
                    "private_case_passes": private["case_passes"],
                    "actual_vm_calls": int(visible["actual_vm_call"])
                    + int(private["actual_vm_call"]),
                    "rejection": visible.get("error"),
                }
            )
        return results
    import numpy as np

    from mindscape.v3.physics_world.data import generate_scene
    from mindscape.v3.physics_world.environment import Environment
    from mindscape.v3.physics_world.schema import Action

    p = task["public"]
    results = []
    for proposal in proposals:
        env = Environment(generate_scene(p["family"], "iid", p["index"], p["seed"]))
        for _ in range(3):
            env.step(Action.from_vector(np.zeros((3, 4))))
        for step in range(p["horizon"]):
            state = env.step(
                Action.from_vector(proposal["action"] if step == 0 else np.zeros((3, 4)))
            )
        distance = float(np.linalg.norm(state[0, :2] - p["target"]))
        results.append(
            {"success": distance <= 0.75, "distance": distance, "terminal": state.tolist()}
        )
    return results


def replay(root):
    from mindscape.v3.schema import digest

    from .data import tasks
    from .encoding import encode

    root = Path(root)
    expected = {t["identity"]: t for t in tasks()}
    rows = [json.loads(line) for line in (root / "rows.jsonl").read_text().splitlines()]
    if len(rows) != len(expected) or {r["identity"] for r in rows} != set(expected):
        raise ValueError("Incomplete or duplicate task coverage")
    for row in rows:
        task = expected[row["identity"]]
        saved = json.loads((root / "proposals" / f"{row['identity']}.json").read_text())
        if digest(saved) != row["proposal_digest"] or any(row[k] != v for k, v in saved.items()):
            raise ValueError("Proposal receipt mismatch")
        if (row["domain"], row["split"]) != (task["domain"], task["split"]):
            raise ValueError("Identity metadata mismatch")
        features = [
            encode(row["domain"], row["public"], p, s)
            for p, s in zip(row["proposals"], row["scores"])
        ]
        if row["features"] != features or outcomes(task, row["proposals"]) != row["outcomes"]:
            raise ValueError("Independent outcome/encoding mismatch")
        if row["labels"] != [float(v["success"]) for v in row["outcomes"]]:
            raise ValueError("Label mismatch")
    receipt = {
        "status": "PASS",
        "tasks": len(rows),
        "candidate_executions": sum(len(r["proposals"]) for r in rows),
        "rows_digest": digest(rows),
    }
    (root / "replay.json").write_text(json.dumps(receipt, indent=2))
    return receipt


if __name__ == "__main__":
    import sys

    print(json.dumps(replay(sys.argv[1])))
