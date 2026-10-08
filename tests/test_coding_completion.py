import json
from pathlib import Path

from mindscape.coding.instrumentation import StageMeter
from mindscape.coding.memory import CodingMemory


def test_stage_meter_restores_methods_and_records_actual_calls():
    original = CodingMemory.retrieve
    with StageMeter() as meter:
        assert CodingMemory().retrieve("repair") == []
        assert meter.summary()["memory_retrieval"]["calls"] == 1
    assert CodingMemory.retrieve is original
    assert meter.summary()["memory_retrieval"]["seconds"] >= 0


def test_locked_completion_tasks_exclude_every_training_seed():
    data_path = Path("results/coding/final_dataset_v1/dataset.json")
    lock = Path("results/coding/completion_lockbox_v1/tasks.json")
    if not data_path.exists() or not lock.exists():
        import pytest

        pytest.skip("Optional research data not installed")
    data = json.loads(data_path.read_text())
    tasks = json.loads(lock.read_text())
    old = {t["metadata"]["seed"] for ts in data.values() for t in ts}
    assert len(tasks) == 100
    assert not old.intersection(t["metadata"]["seed"] for t in tasks)
    assert {
        t["metadata"]["structural_category"] for t in tasks if t["metadata"]["split"] == "ood_test"
    }.isdisjoint({t["metadata"]["structural_category"] for t in data["train"]})


def test_teacher_memory_rejects_heldout_ground_truth_before_execution():
    import pytest

    from mindscape.coding.generator import task
    from mindscape.coding.policy import teacher_memory

    with pytest.raises(ValueError, match="training tasks only"):
        teacher_memory([task(12345, "operator", "test")], None, 1)


def test_bounded_planner_never_certifies_hypothetical_tests():
    from mindscape.coding.environment import CodeRepairEnvironment
    from mindscape.coding.generator import task
    from mindscape.coding.planning import BoundedPlanner
    from mindscape.coding.schema import CodeAction

    env = CodeRepairEnvironment(None)
    state = env.reset(task(12345, "operator", "test"))
    path = next(iter(state.files))
    action, records = BoundedPlanner().plan(
        state, CodeAction("edit", path=path, content="def broken(:\n")
    )
    assert action.name == "run_tests"
    assert len(records) == 2
    assert all(
        r["predicted_state"]["hypothetical"] and r["actual_tests_passed"] is None for r in records
    )
    assert env.files == state.files
    env.close()


def test_perturbations_preserve_actual_correct_and_defective_behavior():
    import pytest

    from mindscape.coding.generator import task
    from mindscape.coding.robustness import perturb
    from mindscape.coding.sandbox import WasiSandbox
    from mindscape.coding.testing import run_cases

    if not Path("work/coding/runtime/python.wasm").exists():
        pytest.skip("Optional pinned WASI runtime absent")
    sandbox = WasiSandbox("work/coding/runtime")
    for category in ("list_handling", "cross_file"):
        original = task(99881, category, "test")
        for variant in (
            "rename_functions",
            "rename_variables",
            "reordered_files",
            "reordered_tests",
            "distractor_files",
            "layout",
            "formatting",
        ):
            changed = perturb(original, variant)
            assert run_cases(sandbox, changed.correct_repository, changed.hidden_tests)[
                "all_passed"
            ]
            assert not run_cases(sandbox, changed.repository, changed.hidden_tests)["all_passed"]


def test_generic_model_wrapper_defers_private_assessment():
    from types import SimpleNamespace

    from mindscape.coding.generator import task
    from mindscape.coding.integration import CodingBenchmarkModel

    example = task(775511, "operator", "test")

    class Policy:
        coder = SimpleNamespace(parameter_count=17)

        def solve(self, learner):
            assert learner.hidden_tests is None
            assert learner.correct_repository == {} and learner.ground_truth_patch == {}
            return {
                "repository": learner.repository,
                "trajectory": {},
                "model_calls": 1,
                "wall_seconds": 0.1,
            }

    prediction = CodingBenchmarkModel(Policy()).predict(
        {
            "example_id": example.task_id,
            "problem": example.problem_statement,
            "observation": example.visible(),
        }
    )
    assert prediction.diagnostics["assessment"] == "deferred"
    assert prediction.answer == example.repository
