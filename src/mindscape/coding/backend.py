"""Coding adapter for the existing generic dataset/evaluation interfaces."""

import os
from dataclasses import asdict

from mindscape.coding.environment import CodeRepairEnvironment
from mindscape.coding.generator import OOD, TRAIN, task
from mindscape.coding.sandbox import WasiSandbox
from mindscape.coding.schema import CodeAction, CodeTask
from mindscape.coding.testing import run_cases
from mindscape.data.backends import register, stable_hash
from mindscape.data.schemas import BenchmarkExample
from mindscape.evaluation.schemas import PredictionRecord


class CodingBackend:
    name = "python_code_repair"
    version = "coding-1.0"

    def structures(self, spec):
        values = spec.get("structures", TRAIN)
        for value in values:
            self.structure_identity(value)
        return list(values)

    def structure_identity(self, category):
        if category not in TRAIN + OOD:
            raise ValueError("Unknown coding structure")
        return category

    def sample(self, category, rng, config):
        self.structure_identity(category)
        return {"seed": rng.randrange(100000, 999999999), "structure": category}

    def identity(self, observation):
        return stable_hash(observation)

    def category(self, observation):
        return observation["structure"]

    def example(self, observation, seed, split):
        t = task(observation["seed"], observation["structure"], split)
        env = CodeRepairEnvironment(None)
        env.reset(t)
        try:
            initial = asdict(env.get_state())
        finally:
            env.close()
        return BenchmarkExample(
            t.task_id,
            self.name,
            t.problem_statement,
            t.visible() | observation,
            initial,
            {"description": "Pass independent terminal private tests"},
            t.correct_repository,
            [],
            t.metadata | {"random_seed": seed, "private_tests": t.hidden_tests},
        )

    def validate(self, example):
        expected = self.example(
            {"seed": example.observation["seed"], "structure": example.observation["structure"]},
            example.metadata["random_seed"],
            example.metadata["split"],
        )
        if stable_hash(expected.to_dict()) != stable_hash(example.to_dict()):
            raise ValueError("Corrupt coding target/schema/metadata")

    def _sandbox(self):
        return WasiSandbox(os.environ.get("MINDSCAPE_WASI_RUNTIME", "work/coding/runtime"))

    def assess(self, example, answer, trajectory):
        if not isinstance(answer, dict) or set(answer) != set(example.observation["repository"]):
            return False, False, "malformed_state"
        sandbox = self._sandbox()
        initial = example.observation["repository"]
        t = CodeTask(
            example.example_id,
            initial,
            example.problem,
            example.observation["visible_tests"],
            example.metadata["private_tests"],
            {},
            {},
            {},
        )
        valid = None
        if trajectory is not None:
            env = CodeRepairEnvironment(sandbox)
            env.reset(t)
            valid = True
            try:
                if trajectory.get("initial_state", {}).get("files") != initial:
                    valid = False
                for record in trajectory.get("transitions", []):
                    actual = env.step(CodeAction(**record["action"]))
                    if (
                        not actual.valid
                        or actual.event.name != record["event"]["name"]
                        or actual.state_before.files != record["state_before"]["files"]
                        or actual.state_after.files != record["state_after"]["files"]
                    ):
                        valid = False
                    if (
                        actual.action.name in ("run_test", "run_tests")
                        and actual.result.stdout != record["result"]["stdout"]
                    ):
                        valid = False
                if env.files != answer:
                    valid = False
            except (ValueError, TypeError, KeyError, AttributeError):
                valid = False
            finally:
                env.close()
        goal = run_cases(sandbox, answer, t.hidden_tests)["all_passed"]
        return (
            valid,
            goal,
            "correct"
            if goal and valid is not False
            else "invalid_transition"
            if valid is False
            else "goal_failure",
        )

    def reference(self, view):
        raise RuntimeError(
            "Coding reference solutions must never be inferred from evaluation views"
        )


register(CodingBackend())


def evaluate_coding(example, prediction, backend):
    valid, goal, error = backend.assess(example, prediction.answer, prediction.trajectory)
    correct = bool(goal)
    grounded = correct and valid is True
    return PredictionRecord(
        example.example_id,
        prediction.answer,
        "independent hidden tests",
        correct,
        prediction.trajectory,
        valid,
        correct,
        grounded,
        error,
        prediction.diagnostics,
    )
