"""Independent outcomes: register interpreter, real WASI subtree evaluation, pinned Pymunk."""

from dataclasses import asdict
from pathlib import Path

from .data import expression_nodes, public_before
from .schema import SCALES


def coding_repository(public, edited):
    values, _, links = expression_nodes(public)
    if edited and 0 <= public["target_leaf"] < len(public["constants"]):
        values[public["target_leaf"] + 1] += public["increment"]
    expr = ["x"] + [repr(v) for v in values[1 : 1 + len(public["constants"])]]
    for _, a, b, op in links:
        symbol = {"add": "+", "sub": "-", "mul": "*"}[op]
        expr.append(f"({expr[a]} {symbol} {expr[b]})")
    return {"solution.py": f"def {public['entry']}(x):\n    return [{', '.join(expr)}]\n"}


def execute(task, executor=None):
    d = task["domain"]
    p = task["public"]
    before = public_before(task)
    if d == "mathematics":
        valid = 0 <= p["destination"] < len(before) and 0 <= p["source"] < len(before)
        after = [r.copy() for r in before]
        if valid:
            amount = p["amount"] * before[p["source"]][0]
            after[p["destination"]][0] += amount if p["operation"] == "add" else -amount
        provenance = {"executor": "bounded_register_interpreter"}
    elif d == "coding":
        from mindscape.v3.coding_learning.execution import Executor

        if executor is None:
            executor = Executor(Path("work/coding/runtime"))
        valid = 0 <= p["target_leaf"] < len(p["constants"])
        suite = {
            "entry": "solution." + p["entry"],
            "cases": [
                {"args": [p["x"]], "expected": [], "expected_type": "list", "unchanged": True}
            ],
        }
        # Expected placeholder never enters program execution; retrieve actual VM values.
        outputs = []
        for edited in (False, True):
            receipt = executor.run(coding_repository(p, edited), suite)
            if (
                not receipt["actual_vm_call"]
                or len(receipt["observed"]) != 1
                or "error" in receipt["observed"][0]
            ):
                raise ValueError("Actual WASI observation required")
            outputs.append([[float(v)] for v in receipt["observed"][0]["value"]])
        before, after = outputs
        if before != public_before(task):
            raise ValueError("Independent coding before-state mismatch")
        # Separate mathematical replay of the explicit AST, not a neural/candidate oracle.
        modified = {**p, "constants": list(p["constants"])}
        if valid:
            modified["constants"][p["target_leaf"]] += p["increment"]
        if after != public_before({"domain": "coding", "public": modified}):
            raise ValueError("WASI/AST independent effect mismatch")
        provenance = {
            "executor": "actual_WASI",
            "calls": 2,
            "repositories": [coding_repository(p, False), coding_repository(p, True)],
        }
    else:
        import numpy as np

        from mindscape.v3.physics_world.environment import Environment
        from mindscape.v3.physics_world.schema import Action, ObjectSpec, Scene

        objects = tuple(
            ObjectSpec(tuple(pos), tuple(vel), mass, 0.25, 0.2, 0.5)
            for pos, vel, mass in zip(p["positions"], p["velocities"], p["masses"])
        )
        scene = Scene(
            "motion",
            "iid",
            0,
            p["scene_seed"],
            "v5_impulse",
            objects,
            p["gravity"],
            0.0,
            False,
            False,
            0.0,
            1,
        )
        env = Environment(scene)
        valid = 0 <= p["destination"] < len(objects)
        if valid:
            action = np.zeros((3, 4))
            action[p["destination"], 2:] = p["impulse"]
            state = env.step(Action.from_vector(action))
            after = state[: len(objects), :4].tolist()
        else:
            after = [
                r.copy() for r in before
            ]  # Explicit invalid-action precondition: reject as no-op.
        provenance = {
            "executor": "pinned_Pymunk",
            "scene": asdict(scene),
            "invalid_policy": "reject_without_step",
        }
    delta = [[(b - a) / SCALES[d] for a, b in zip(x, y)] for x, y in zip(before, after)]
    return {
        "valid": int(valid),
        "before": before,
        "after": after,
        "delta": delta,
        "provenance": provenance,
    }
