"""Immutable, machine-readable cognitive records."""
from dataclasses import dataclass
from enum import Enum


class EvidenceKind(str, Enum):
    OBSERVATION = "observation"
    INFERENCE = "inference"
    PREDICTION = "prediction"
    ACTUAL_RESULT = "actual_result"
    HYPOTHETICAL_RESULT = "hypothetical_result"


@dataclass(frozen=True)
class Entity:
    id: str
    value: int


@dataclass(frozen=True)
class Relation:
    source: str
    predicate: str
    target: str


@dataclass(frozen=True)
class Observation:
    operands: tuple[int, int]
    source: str = "user_input"
    kind: EvidenceKind = EvidenceKind.OBSERVATION

    def __post_init__(self):
        if (type(self.operands) is not tuple or len(self.operands) != 2
                or any(type(x) is not int for x in self.operands)):
            raise ValueError("Exactly two integer operands are required")
        if self.kind != EvidenceKind.OBSERVATION:
            raise ValueError("Input must be an observation")


@dataclass(frozen=True)
class Goal:
    description: str = "Produce the correct signed integer product"


@dataclass(frozen=True)
class State:
    entities: tuple[Entity, ...]
    relations: tuple[Relation, ...]
    left_digits: tuple[int, ...]
    right_digits: tuple[int, ...]
    sign: int
    position: int = 0
    row_position: int = 0
    carry: int = 0
    row_value: int = 0
    accumulated: int = 0
    phase: str = "multiply"
    answer: int | None = None
    context: str = "decimal_integer_multiplication"
    goal: Goal = Goal()


@dataclass(frozen=True)
class Action:
    name: str
    position: int
    row_position: int


@dataclass(frozen=True)
class Event:
    name: str
    detail: str


@dataclass(frozen=True)
class Result:
    value: int
    written_digit: int | None = None
    carry: int = 0
    kind: EvidenceKind = EvidenceKind.ACTUAL_RESULT
    source: str = "multiplication_environment"


@dataclass(frozen=True)
class Transition:
    state_before: State
    action: Action
    event: Event
    result: Result
    state_after: State


@dataclass(frozen=True)
class Trajectory:
    observation: Observation
    initial_state: State
    transitions: tuple[Transition, ...] = ()
