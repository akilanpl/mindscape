"""Real isolated repository transitions; hidden evaluation is never a policy action."""

import ast
import copy
import difflib
import json
import tempfile
import time
from dataclasses import asdict
from pathlib import Path, PurePosixPath

from mindscape.coding.schema import (
    CodeAction,
    CodeEvent,
    CodeResult,
    CodeState,
    CodeTrajectory,
    CodeTransition,
)
from mindscape.coding.testing import run_cases
from mindscape.core.schema import Entity, Goal, Relation

ACTIONS = (
    "inspect_repo",
    "inspect_file",
    "inspect_symbol",
    "search",
    "inspect_test",
    "edit",
    "patch",
    "run_test",
    "run_tests",
    "inspect_error",
    "revert",
    "compare_versions",
    "finish",
)


class CodeRepairEnvironment:
    def __init__(self, sandbox):
        self.sandbox = sandbox
        self._tmp = None

    def reset(self, task):
        if self._tmp:
            self._tmp.cleanup()
        self._tmp = tempfile.TemporaryDirectory(prefix="mindscape-repository-")
        self.root = Path(self._tmp.name)
        self.task = task
        self.original = dict(task.repository)
        self.files = dict(task.repository)
        self.transitions = []
        self.test_status = None
        self.previous = []
        self.observations = []
        self.errors = []
        self._write()
        self.initial = self.get_state()
        return self.initial

    def _write(self):
        for name, content in self.files.items():
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or not name.endswith(".py"):
                raise ValueError("Unsafe repository path")
            p = self.root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)

    def get_goal(self):
        return Goal("Independent terminal evaluation: all hidden tests pass")

    def valid_actions(self):
        return tuple(CodeAction(name) for name in ACTIONS)

    def get_observation(self):
        return {
            "repository": dict(self.files),
            "visible_tests": self.task.visible_tests,
            "task": self.task.problem_statement,
            "actual_observations": copy.deepcopy(self.observations),
        }

    def get_state(self):
        symbols = {}
        relations = []
        entities = []
        for index, (path, source) in enumerate(sorted(self.files.items())):
            entities.append(Entity(path, index))
            try:
                tree = ast.parse(source)
                nodes = [
                    n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                ]
                symbols[path] = tuple(n.name for n in nodes)
                for n in nodes:
                    identity = path + ":" + n.name
                    entities.append(Entity(identity, index))
                    relations.append(Relation(path, "CONTAINS", identity))
                    for call in ast.walk(n):
                        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
                            relations.append(Relation(identity, "CALLS", call.func.id))
            except SyntaxError:
                symbols[path] = ()
                relations.append(Relation("syntax_error", "ORIGINATES_IN", path))
        spec = json.loads(self.task.visible_tests)
        target = spec["entry"]
        passes = (self.test_status or {}).get("case_passes", [])
        for i in range(len(spec["cases"])):
            passed = passes[i] if i < len(passes) else None
            identity = "visible_test_" + str(i)
            entities.append(Entity(identity, i))
            relations.append(Relation(identity, "TARGETS", target))
            if passed is False:
                relations.append(Relation(identity, "FAILS_BECAUSE_OF", "visible_failure"))
        changed = tuple(p for p in self.files if self.files[p] != self.original[p])
        for p in changed:
            relations.append(Relation("patch", "MODIFIES", p))
        return CodeState(
            self.task.task_id,
            tuple(sorted(self.files)),
            dict(self.files),
            symbols,
            changed,
            tuple(str(i) for i, p in enumerate(passes) if not p),
            tuple(str(i) for i, p in enumerate(passes) if p),
            tuple(self.errors),
            tuple(copy.deepcopy(self.observations)),
            self.previous[-1] if self.previous else None,
            tuple(self.previous),
            self.get_goal(),
            "visible_pass"
            if passes and all(passes)
            else "unverified"
            if not passes
            else "visible_failure",
            {
                "problem_statement": self.task.problem_statement,
                "visible_tests": self.task.visible_tests,
                "tools": ACTIONS,
            },
            tuple(entities),
            tuple(relations),
        )

    def inspect(self, path):
        return self.step(CodeAction("inspect_file", path=path))

    def get_trajectory(self):
        return CodeTrajectory(self.task.task_id, self.initial, tuple(self.transitions))

    def step(self, action):
        if not isinstance(action, CodeAction):
            raise TypeError("CodeAction required")
        before = self.get_state()
        began = time.perf_counter()
        valid = True
        name = "observed"
        stdout = ""
        stderr = ""
        changed = ()
        returncode = None
        try:
            if not isinstance(action, CodeAction) or action.name not in ACTIONS:
                raise ValueError("Unknown typed action")
            if action.path is not None and action.path not in self.files:
                raise ValueError("Path is outside this task repository")
            if action.name == "inspect_repo":
                stdout = json.dumps({"files": sorted(self.files), "symbols": before.symbols})
            elif action.name == "inspect_file":
                if action.path is None:
                    raise ValueError("File path required")
                stdout = self.files[action.path]
            elif action.name == "inspect_symbol":
                if not action.symbol:
                    raise ValueError("Symbol required")
                found = []
                for p, s in self.files.items():
                    for n in ast.walk(ast.parse(s)):
                        if (
                            isinstance(n, (ast.FunctionDef, ast.ClassDef))
                            and n.name == action.symbol
                        ):
                            found.append({"path": p, "source": ast.get_source_segment(s, n)})
                stdout = json.dumps(found)
            elif action.name == "search":
                if not action.query:
                    raise ValueError("Query required")
                stdout = json.dumps(
                    [
                        {"path": p, "line": i + 1, "text": line}
                        for p, s in self.files.items()
                        for i, line in enumerate(s.splitlines())
                        if action.query in line
                    ]
                )
            elif action.name == "inspect_test":
                stdout = self.task.visible_tests
            elif action.name in ("edit", "patch"):
                if not action.path:
                    raise ValueError("Path required")
                if action.name == "edit":
                    if action.content is None or len(action.content) > 64000:
                        raise ValueError("Bounded source content required")
                    source = action.content
                else:
                    if (
                        not action.old
                        or action.new is None
                        or self.files[action.path].count(action.old) != 1
                    ):
                        raise ValueError("Patch must uniquely match current source")
                    source = self.files[action.path].replace(action.old, action.new, 1)
                if len(source) > 64000:
                    raise ValueError("Source exceeds limit")
                self.files[action.path] = source
                self._write()
                self.test_status = None
                changed = (action.path,)
                name = "file_modified"
            elif action.name in ("run_test", "run_tests"):
                suite = self.task.visible_tests
                if action.name == "run_test":
                    spec = json.loads(suite)
                    index = int(action.test or "0")
                    if not 0 <= index < len(spec["cases"]):
                        raise ValueError("Unknown visible test")
                    spec["cases"] = [spec["cases"][index]]
                    suite = json.dumps(spec)
                outcome = run_cases(self.sandbox, self.files, suite)
                self.test_status = outcome
                stdout = json.dumps({k: v for k, v in outcome.items() if k != "execution"})
                stderr = outcome["execution"]["stderr"]
                returncode = outcome["execution"]["returncode"]
                name = (
                    "timeout"
                    if outcome["execution"]["timeout"]
                    else "test_passed"
                    if outcome["all_passed"]
                    else "syntax_error"
                    if "SyntaxError" in stderr
                    else "test_failed"
                )
                if not outcome["all_passed"]:
                    self.errors.append(stderr or stdout)
            elif action.name == "inspect_error":
                stdout = json.dumps(self.errors[-3:])
            elif action.name == "revert":
                paths = [action.path] if action.path else list(self.files)
                for p in paths:
                    self.files[p] = self.original[p]
                self._write()
                self.test_status = None
                changed = tuple(paths)
                name = "file_reverted"
            elif action.name == "compare_versions":
                stdout = "\n".join(
                    "".join(
                        difflib.unified_diff(
                            self.original[p].splitlines(True),
                            self.files[p].splitlines(True),
                            fromfile="initial/" + p,
                            tofile="current/" + p,
                        )
                    )
                    for p in sorted(self.files)
                )
            elif action.name == "finish":
                name = "finish_requested"
        except (ValueError, TypeError, SyntaxError) as error:
            valid = False
            name = "patch_rejected" if action.name in ("edit", "patch") else "invalid_action"
            stderr = str(error)
        result = CodeResult(stdout, stderr, returncode, time.perf_counter() - began, changed)
        self.previous.append(action.name)
        self.observations.append(
            {"action": asdict(action), "event": name, "result": asdict(result)}
        )
        after = self.get_state()
        transition = CodeTransition(before, action, CodeEvent(name), result, after, valid)
        self.transitions.append(transition)
        return transition

    def final_evaluate(self):
        # Host-authoritative private inputs/expected values; never returned in model observations.
        actual = {path: (self.root / path).read_text() for path in self.files}
        if actual != self.files:
            raise RuntimeError("Repository state differs from files on disk")
        return run_cases(self.sandbox, actual, self.task.hidden_tests)

    def is_goal_reached(self):
        return self.final_evaluate()["all_passed"]

    def close(self):
        if self._tmp:
            self._tmp.cleanup()
            self._tmp = None
