"""Actual coding LM policies; adaptive memory is not gradient fine-tuning."""

import json
import time
from dataclasses import asdict
from pathlib import Path

from mindscape.coding.environment import CodeRepairEnvironment
from mindscape.coding.memory import CodingMemory, words
from mindscape.coding.model import edit_from_response
from mindscape.coding.planning import BoundedPlanner
from mindscape.coding.schema import CodeAction


class CodingPolicy:
    def __init__(
        self, coder, sandbox, condition="model_only", memory=None, max_attempts=3, ablation=None
    ):
        self.coder = coder
        self.sandbox = sandbox
        self.condition = condition
        self.memory = memory or CodingMemory()
        self.max_attempts = max_attempts
        self.ablation = ablation

    def solve(self, task, seed=0):
        if self.condition == "experiential":
            return ToolCodingPolicy(
                self.coder, self.sandbox, self.memory, ablation=self.ablation
            ).solve(task, seed)
        env = CodeRepairEnvironment(self.sandbox)
        env.reset(task)
        self.memory.reset_working()
        start = time.perf_counter()
        begin_calls = self.coder.calls
        attempts = []
        dreams = []
        interaction = (
            self.condition == "experiential" and self.ablation != "no_environment_feedback"
        )
        structured = self.condition != "model_only" and self.ablation != "no_state"
        try:
            if structured:
                env.step(CodeAction("inspect_repo"))
                env.step(CodeAction("inspect_test"))
            if interaction:
                env.step(CodeAction("run_tests"))
            exemplars = (
                []
                if self.ablation == "no_memory" or self.condition in ("model_only", "structured")
                else self.memory.retrieve(task.problem_statement)
            )
            for index in range(self.max_attempts if interaction else 1):
                state = env.get_state()
                payload = _model_view(env.get_observation())
                if structured:
                    payload["state"] = {
                        "symbols": state.symbols,
                        "changes": state.current_changes,
                        "progress": state.progress,
                        "failures": state.failing_tests,
                    }
                    if self.ablation != "no_relation_graph":
                        payload["relations"] = [asdict(x) for x in state.relations]
                    if self.ablation != "no_goal":
                        payload["goal"] = state.goal.description
                if exemplars:
                    payload["training_example"] = exemplars[0]
                    if self.ablation == "no_trajectory":
                        payload["training_example"] = {
                            k: v for k, v in exemplars[0].items() if k != "trajectory"
                        }
                if interaction and self.ablation != "no_dream":
                    dreams = self.memory.dream(
                        state,
                        [
                            {"name": "edit", "reason": "propose a source repair"},
                            {"name": "revert", "reason": "discard a regression"},
                        ],
                    )
                    payload["hypothetical_options"] = dreams
                payload["instruction"] = (
                    "Select the file that needs repair and return one JSON object with path and content containing its complete corrected Python source. Do not include hidden tests."
                )
                response = self.coder.generate(
                    "You are a Python repair agent. Choose a concrete source edit using the supplied evidence. Return JSON only.",
                    json.dumps(payload),
                    256,
                    seed,
                )
                error = None
                try:
                    edit = edit_from_response(
                        response, env.files, json.loads(task.visible_tests)["entry"]
                    )
                    transition = env.step(edit)
                    if not transition.valid:
                        raise ValueError(transition.result.stderr)
                except (ValueError, TypeError, AttributeError, json.JSONDecodeError) as exc:
                    error = str(exc)
                visible = None
                if interaction:
                    visible = env.step(CodeAction("run_tests")).event.name == "test_passed"
                attempts.append(
                    {
                        "response": response,
                        "parse_error": error,
                        "visible_pass": visible,
                        "latency": dict(self.coder.timings[-1]),
                    }
                )
                self.memory.remember(
                    {
                        "problem": task.problem_statement,
                        "attempt": index,
                        "visible_pass": visible,
                        "response": response,
                    },
                    verified=False,
                )
                if visible:
                    break
                if interaction:
                    env.step(CodeAction("inspect_error"))
            env.step(CodeAction("finish"))
            final = env.final_evaluate()
            return {
                "task_id": task.task_id,
                "condition": self.condition,
                "ablation": self.ablation,
                "success": final["all_passed"],
                "attempts": attempts,
                "repository": dict(env.files),
                "trajectory": asdict(env.get_trajectory()),
                "terminal": final,
                "model_calls": self.coder.calls - begin_calls,
                "wall_seconds": time.perf_counter() - start,
                "dreams": dreams,
            }
        finally:
            env.close()


def teacher_memory(tasks, sandbox, limit):
    """Actual expert trajectories, executed and checked only on training tasks."""
    memory = CodingMemory(capacity=limit)
    for task in tasks[:limit]:
        if task.metadata.get("split") != "train":
            raise ValueError("Teacher memory accepts training tasks only")
        env = CodeRepairEnvironment(sandbox)
        env.reset(task)
        try:
            env.step(CodeAction("inspect_repo"))
            env.step(CodeAction("inspect_test"))
            env.step(CodeAction("run_tests"))
            for path, content in task.ground_truth_patch.items():
                env.step(CodeAction("edit", path=path, content=content))
            verified = env.step(CodeAction("run_tests")).event.name == "test_passed"
            if not verified:
                raise RuntimeError(
                    "Teacher repair failed actual visible execution: " + task.task_id
                )
            trace_root = Path("results/coding/teacher_traces_v1")
            trace_root.mkdir(parents=True, exist_ok=True)
            trace_path = trace_root / (task.task_id + ".json")
            if not trace_path.exists():
                trace_path.write_text(
                    json.dumps(
                        {
                            "task_id": task.task_id,
                            "trajectory": asdict(env.get_trajectory()),
                            "visible_verified": verified,
                        }
                    )
                )
            memory.remember(
                {
                    "problem": task.problem_statement,
                    "initial_repository": task.repository,
                    "patch": task.ground_truth_patch,
                    "trajectory": [
                        {"action": asdict(t.action), "event": t.event.name, "valid": t.valid}
                        for t in env.get_trajectory().transitions
                    ],
                },
                verified=verified,
            )
        finally:
            env.close()
    return memory


class ToolCodingPolicy:
    """The coding LM selects typed tools, including inspection, edits and stopping."""

    def __init__(self, coder, sandbox, memory, max_steps=8, max_edits=3, ablation=None):
        self.coder = coder
        self.sandbox = sandbox
        self.memory = memory
        self.max_steps = max_steps
        self.max_edits = max_edits
        self.ablation = ablation

    def solve(self, task, seed=0):
        from mindscape.coding.model import action_from_response
        from mindscape.coding.testing import run_cases

        env = CodeRepairEnvironment(self.sandbox)
        env.reset(task)
        self.memory.reset_working()
        start = time.perf_counter()
        begin = self.coder.calls
        attempts = []
        snapshots = []
        dreams = []
        try:
            for step in range(self.max_steps):
                state = env.get_state()
                payload = _model_view(env.get_observation())
                payload["tools"] = list(state.context["tools"])
                payload["remaining_steps"] = self.max_steps - step
                payload["controller_errors"] = [
                    a["parse_error"] for a in attempts if a["parse_error"]
                ][-2:]
                if self.ablation != "no_memory":
                    payload["working_memory"] = self.memory.working[-2:]
                if self.ablation != "no_state":
                    payload["state"] = {
                        "symbols": state.symbols,
                        "changes": state.current_changes,
                        "progress": state.progress,
                        "previous_actions": state.previous_actions,
                    }
                if self.ablation != "no_relation_graph":
                    payload["relations"] = [asdict(r) for r in state.relations]
                if self.ablation != "no_goal":
                    payload["goal"] = state.goal.description
                if self.ablation != "no_memory":
                    examples = self.memory.retrieve(task.problem_statement)
                    if examples:
                        payload["executed_training_example"] = dict(examples[0])
                        payload["concept_memory"] = {
                            "retrieved_problem": examples[0]["problem"],
                            "evidence": "executed training repair",
                            "concept": self.memory.concepts.get(
                                " ".join(sorted(words(examples[0]["problem"]))), {}
                            ),
                        }
                        if self.ablation == "no_trajectory":
                            payload["executed_training_example"].pop("trajectory", None)
                if self.ablation != "no_dream":
                    dreams = self.memory.dream(state, [{"name": "run_tests"}, {"name": "edit"}])
                    payload["hypothetical_options"] = dreams
                payload["instruction"] = (
                    "Choose exactly one tool action as JSON. Fields: name, optional path/symbol/query/content/old/new/test. For edit, content is the complete corrected Python file. Use actual visible tests to verify changes; finish when satisfied. Inspection and test actions are available; private evaluation is inaccessible."
                )
                response = self.coder.generate(
                    "You are a bounded Python repair agent. Select and execute one tool action per turn. Output a single JSON action, no explanation.",
                    json.dumps(payload),
                    256,
                    seed,
                )
                error = None
                action = None
                predictions = []
                transition = None
                try:
                    action = action_from_response(response)
                    if self.ablation != "no_dream":
                        action, predictions = BoundedPlanner().plan(state, action)
                        dreams = predictions
                    if action.name in ("edit", "patch") and len(snapshots) >= self.max_edits:
                        raise ValueError("Edit budget exhausted")
                    if self.ablation == "no_environment_feedback" and action.name in (
                        "run_test",
                        "run_tests",
                        "inspect_error",
                    ):
                        raise ValueError("Execution feedback removed by ablation")
                    transition = env.step(action)
                    if action.name in ("edit", "patch") and transition.valid:
                        snapshots.append(dict(env.files))
                    if not transition.valid:
                        error = transition.result.stderr
                except (
                    ValueError,
                    TypeError,
                    AttributeError,
                    KeyError,
                    json.JSONDecodeError,
                ) as exc:
                    error = str(exc)
                attempts.append(
                    {
                        "response": response,
                        "parse_error": error,
                        "action": asdict(action) if action else None,
                        "planning": predictions,
                        "visible_pass": transition.event.name == "test_passed"
                        if transition
                        else None,
                        "latency": dict(self.coder.timings[-1]),
                    }
                )
                if transition:
                    self.memory.remember(
                        {
                            "action": asdict(action),
                            "event": transition.event.name,
                            "valid": transition.valid,
                        },
                        verified=False,
                    )
                if action and action.name == "finish" and transition and transition.valid:
                    break
            final = env.final_evaluate()
            first = (
                run_cases(self.sandbox, snapshots[0], task.hidden_tests)["all_passed"]
                if snapshots
                else False
            )
            return {
                "task_id": task.task_id,
                "condition": "experiential",
                "ablation": self.ablation,
                "success": final["all_passed"],
                "first_patch_success": first,
                "edit_attempts": len(snapshots),
                "attempts": attempts,
                "repository": dict(env.files),
                "trajectory": asdict(env.get_trajectory()),
                "terminal": final,
                "model_calls": self.coder.calls - begin,
                "wall_seconds": time.perf_counter() - start,
                "dreams": dreams,
            }
        finally:
            env.close()


def _model_view(value):
    """Exclude timing metadata from policy inputs; retain it in actual transcripts."""
    if isinstance(value, dict):
        return {
            k: _model_view(v)
            for k, v in value.items()
            if k not in ("duration", "generation_seconds", "ttft")
        }
    if isinstance(value, (tuple, list)):
        return [_model_view(v) for v in value]
    return value
