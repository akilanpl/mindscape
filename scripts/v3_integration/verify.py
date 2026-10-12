"""Independent CPU arithmetic, WASI and physics replay; no neural execution."""

import argparse
import json
from pathlib import Path

import numpy as np

from mindscape.v3.integration.runtime import Request
from mindscape.v3.kernel import BoundedPolicy
from mindscape.v3.physics_world.storage import atomic_json, file_sha
from mindscape.v3.schema import Action as KernelAction
from mindscape.v3.schema import digest


def verify(root, output):
    root = Path(root)
    frozen = json.loads(Path("configs/v3_integration/requests.json").read_text())
    private = json.loads(Path("configs/v3_integration/private.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    assert summary["suite_hash"] == digest(frozen["requests"])
    assert summary["device"] == "mps"
    assert summary["source_hash"] == digest(
        {str(p): file_sha(p) for p in Path("src/mindscape/v3/integration").resolve().glob("*.py")}
    )
    expected_ids = {Request(**r).identity for r in frozen["requests"]}
    seen = set()
    counts = {}
    for path in root.glob("*-*.json"):
        if path.name.endswith("failure.json"):
            continue
        saved = json.loads(path.read_text())
        if "request" not in saved:
            continue
        request = Request(**saved["request"])
        assert request.identity in expected_ids and request.identity not in seen
        seen.add(request.identity)
        for name in ("direct", "unified"):
            result = saved[name]
            pred = result["prediction"]
            actual = result["actual"]
            assert result["request_id"] == request.identity
            assert pred["kind"] == "hypothetical" and actual["kind"] == "actual"
            if request.domain == "mathematics":
                from mindscape.v3.math_learning.environment import (
                    Claim,
                    LocalState,
                    Problem,
                    execute_local,
                    expected,
                )

                writes = [
                    execute_local(LocalState(**s["state"]), Claim(**s["claim"]))
                    for s in pred["steps"]
                ]
                assert (
                    writes == actual["written_transitions"] and actual["answer"] == pred["answer"]
                )
                problem = Problem(
                    tuple(request.inputs["operands"]),
                    tuple(request.inputs["operations"]),
                    "unit",
                    "integration",
                )
                success = actual["answer"] == str(expected(problem))
            elif request.domain == "coding":
                from mindscape.v3.coding_learning.execution import Executor

                executor = Executor("work/coding/runtime")
                actual_visible = executor.run(actual["repository"], request.inputs["visible_tests"])
                judged = executor.run(actual["repository"], private[request.identity])
                assert actual_visible["all_passed"] == actual["visible"]["all_passed"]
                assert judged["all_passed"] == actual["private_after_termination"]["all_passed"]
                assert actual_visible["actual_vm_call"] and judged["actual_vm_call"]
                success = judged["all_passed"]
            else:
                from mindscape.v3.physics_world.data import generate_scene
                from mindscape.v3.physics_world.environment import Environment
                from mindscape.v3.physics_world.schema import Action

                x = request.inputs
                env = Environment(generate_scene(x["family"], "iid", x["index"], x["seed"]))
                for _ in range(3):
                    env.step(Action.from_vector(np.zeros((3, 4))))
                states = []
                for step in range(x["horizon"]):
                    states.append(
                        env.step(
                            Action.from_vector(actual["action"] if step == 0 else np.zeros((3, 4)))
                        ).copy()
                    )
                np.testing.assert_array_equal(states[-1], actual["terminal"])
                states = np.asarray(states)
                active = states[0, :, 9] > 0
                error = float(
                    (
                        (np.asarray(pred["prediction"])[:, active] - states[:, active, :4]) ** 2
                    ).mean()
                )
                assert np.isclose(error, actual["prediction_mse"])
                success = float(np.linalg.norm(states[-1, 0, :2] - x["target"])) <= 0.75
            assert bool(success) == result["evaluation"]["success"]
            counts[request.domain] = counts.get(request.domain, 0) + 1
        trace = saved["unified"]["trace"]
        assert [e["stage"] for e in trace["events"]] == [
            "OBSERVE",
            "REPRESENT",
            "PROPOSE",
            "PREDICT",
            "ACT",
            "VERIFY",
            "UPDATE",
        ]
        assert trace["events"][-1]["value"]["training_memory_key"] is None
        assert trace["events"][-1]["value"]["weights_updated"] is False
        assert trace["provenance"]["task_id"] == request.identity
        assert trace["provenance"]["source_hash"] == summary["source_hash"]
        assert trace["parameter_count"] > 0 and trace["model_calls"] > 0
        proposed = trace["events"][2]["value"]
        selected, _ = BoundedPolicy().choose(
            proposed["logits"], tuple(KernelAction(**a) for a in proposed["candidates"])
        )
        assert proposed["action"] == {"name": selected.name, "value": selected.value}
        assert trace["events"][3]["value"]["action"] == proposed["action"]
        assert trace["events"][4]["value"]["action"] == proposed["action"]
        assert trace["events"][5]["value"] == saved["unified"]["evaluation"]
        assert trace["persistent_memory_updated"] is False
        if request.domain == "coding":
            assert "private_after_termination" not in json.loads(
                trace["events"][4]["value"]["event"]
            )
        else:
            assert saved["direct"]["prediction"] == saved["unified"]["prediction"]
            assert saved["direct"]["actual"] == saved["unified"]["actual"]
        assert saved["equivalent"] is True
    assert seen == expected_ids, "Missing frozen requests"
    for path, sha in summary["asset_hashes"].items():
        assert file_sha(path) == sha
    receipt = {
        "verified_direct_and_unified_outcomes": counts,
        "neural_calls": 0,
        "engineering_integration": summary["engineering_integration"],
        "learned_cross_domain_transfer": "NOT YET ESTABLISHED",
    }
    atomic_json(output, receipt)
    print(json.dumps(receipt))
    return receipt


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    verify(a.run, a.output)
