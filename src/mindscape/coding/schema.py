from dataclasses import dataclass
from typing import Any

from mindscape.core.schema import Entity, Goal, Relation


@dataclass(frozen=True)
class CodeTask:
    task_id: str
    repository: dict[str, str]
    problem_statement: str
    visible_tests: str
    hidden_tests: str
    correct_repository: dict[str, str]
    ground_truth_patch: dict[str, str]
    metadata: dict[str, Any]

    def visible(self):
        return {
            "task_id": self.task_id,
            "repository": dict(self.repository),
            "problem_statement": self.problem_statement,
            "visible_tests": self.visible_tests,
            "goal": "Pass the task specification; private evaluation is terminal only.",
        }


@dataclass(frozen=True)
class CodeAction:
    name: str
    path: str | None = None
    symbol: str | None = None
    query: str | None = None
    content: str | None = None
    old: str | None = None
    new: str | None = None
    test: str | None = None


@dataclass(frozen=True)
class CodeState:
    task_id: str
    repository: tuple[str, ...]
    files: dict[str, str]
    symbols: dict[str, tuple[str, ...]]
    current_changes: tuple[str, ...]
    failing_tests: tuple[str, ...]
    passing_tests: tuple[str, ...]
    errors: tuple[str, ...]
    observations: tuple[dict, ...]
    current_action: str | None
    previous_actions: tuple[str, ...]
    goal: Goal
    progress: str
    context: dict
    entities: tuple[Entity, ...] = ()
    relations: tuple[Relation, ...] = ()
    hypothetical: bool = False


@dataclass(frozen=True)
class CodeResult:
    stdout: str = ""
    stderr: str = ""
    returncode: int | None = None
    duration: float = 0
    changed_files: tuple[str, ...] = ()
    evidence_kind: str = "actual_result"
    hypothetical: bool = False


@dataclass(frozen=True)
class CodeEvent:
    name: str
    detail: str = ""


@dataclass(frozen=True)
class CodeTransition:
    state_before: CodeState
    action: CodeAction
    event: CodeEvent
    result: CodeResult
    state_after: CodeState
    valid: bool = True


@dataclass(frozen=True)
class CodeTrajectory:
    task_id: str
    initial_state: CodeState
    transitions: tuple[CodeTransition, ...] = ()
