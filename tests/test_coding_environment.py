from pathlib import Path

import pytest

from mindscape.coding.environment import CodeRepairEnvironment
from mindscape.coding.generator import OOD, TRAIN, task
from mindscape.coding.sandbox import WasiSandbox
from mindscape.coding.schema import CodeAction
from mindscape.coding.testing import run_cases


@pytest.fixture(scope="module")
def sandbox():
    runtime = Path("work/coding/runtime")
    if not (runtime / "python.wasm").exists():
        pytest.skip("Optional pinned WASI runtime not installed")
    return WasiSandbox(runtime)


@pytest.mark.parametrize("category", TRAIN + OOD)
def test_independent_task_validation(sandbox, category):
    t = task(19001, category, "validation")
    assert run_cases(sandbox, t.correct_repository, t.hidden_tests)["all_passed"]
    assert not run_cases(sandbox, t.repository, t.hidden_tests)["all_passed"]


def test_real_transitions_and_private_terminal(sandbox):
    t = task(19001, "operator", "test")
    env = CodeRepairEnvironment(sandbox)
    initial = env.reset(t)
    path = next(iter(t.repository))
    assert len(env.valid_actions()) > 1
    assert "hidden_tests" not in env.get_observation()
    assert "correct_repository" not in t.visible()
    assert env.step(CodeAction("run_tests")).event.name == "test_failed"
    assert not env.step(CodeAction("patch", path="../secret", old="x", new="y")).valid
    transition = env.step(CodeAction("edit", path=path, content=t.correct_repository[path]))
    assert transition.state_before.files != transition.state_after.files
    assert initial.files == t.repository
    assert (env.root / path).read_text() == t.correct_repository[path]
    assert env.step(CodeAction("run_tests")).event.name == "test_passed"
    assert env.final_evaluate()["all_passed"]
    assert env.step(CodeAction("compare_versions")).result.stdout
    env.step(CodeAction("revert"))
    assert not env.final_evaluate()["all_passed"]
    assert len(env.get_trajectory().transitions) == 6
    env.close()


def test_capability_isolation(sandbox):
    assert all(sandbox.probe().values())
    result = sandbox.execute({"main.py": "while True: pass\n"}, "import main")
    assert result.returncode != 0


def test_external_mutation_cannot_fake_state(sandbox):
    t = task(19001, "operator", "test")
    env = CodeRepairEnvironment(sandbox)
    env.reset(t)
    (env.root / next(iter(t.repository))).write_text("pass\n")
    with pytest.raises(RuntimeError):
        env.final_evaluate()
    env.close()


def test_shared_backend_uses_execution_not_patch_equality(sandbox):
    from dataclasses import asdict

    from mindscape.coding.backend import evaluate_coding
    from mindscape.data.backends import get_backend
    from mindscape.models.base import Prediction

    backend = get_backend("python_code_repair")
    example = backend.example({"seed": 19001, "structure": "operator"}, 11, "test")
    backend.validate(example)
    view = example.view("experiential")
    assert "private_tests" not in str(view)
    assert "target_answer" not in view
    task_data = task(19001, "operator", "test")
    env = CodeRepairEnvironment(sandbox)
    env.reset(task_data)
    path = next(iter(task_data.repository))
    alternative = task_data.correct_repository[path] + "\n# Another correct implementation\n"
    env.step(CodeAction("edit", path=path, content=alternative))
    trajectory = asdict(env.get_trajectory())
    prediction = Prediction(answer=dict(env.files), trajectory=trajectory)
    record = evaluate_coding(example, prediction, backend)
    assert prediction.answer != example.target_answer
    assert record.correct and record.grounded
    trajectory["transitions"][0]["state_after"]["files"][path] = "pass\n"
    assert not backend.assess(example, prediction.answer, trajectory)[0]
    env.close()


def test_hypothetical_memory_never_becomes_experience():
    from mindscape.coding.memory import CodingMemory

    memory = CodingMemory()
    memory.remember(
        {"problem": "reverse a list", "patch": "fake"}, verified=True, hypothetical=True
    )
    assert not memory.retrieve("reverse a list")
    memory.remember({"problem": "reverse a list", "patch": "actual"}, verified=True)
    assert memory.retrieve("reverse a list")[0]["patch"] == "actual"
