"""V3 records: JSON only, explicit evidence and provenance, no executable codecs."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class Entity:
    id: str
    attributes: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Relationship:
    source: str
    predicate: str
    target: str


@dataclass(frozen=True)
class Concept:
    id: str
    domain: str
    description: str
    relationships: tuple[Relationship, ...] = ()


@dataclass(frozen=True)
class Goal:
    description: str


@dataclass(frozen=True)
class Observation:
    domain: str
    values: tuple[float, ...]
    goal: Goal
    entities: tuple[Entity, ...] = ()
    relationships: tuple[Relationship, ...] = ()
    partial: bool = False
    missing: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.domain or any(not math.isfinite(x) for x in self.values):
            raise ValueError("Finite observations and a domain are required")
        if self.missing and not self.partial:
            raise ValueError("Missing fields require a partial observation")


@dataclass(frozen=True)
class State:
    observation: Observation
    written: tuple[float, ...] = ()
    terminal: bool = False


@dataclass(frozen=True)
class Action:
    name: str
    value: int

    def __post_init__(self):
        if type(self.name) is not str or not self.name or type(self.value) is not int:
            raise ValueError("Action requires a nonempty name and integer value")


@dataclass(frozen=True)
class Prediction:
    action: Action
    state: State
    confidence: float
    kind: str = "hypothetical"

    def __post_init__(self):
        if self.kind != "hypothetical" or not 0 <= self.confidence <= 1:
            raise ValueError("Predictions must be hypothetical with bounded confidence")


@dataclass(frozen=True)
class ActualTransition:
    before: State
    action: Action
    after: State
    event: str
    kind: str = "actual"

    def __post_init__(self):
        if self.kind != "actual":
            raise ValueError("Actual evidence required")


@dataclass(frozen=True)
class Evaluation:
    valid: bool
    success: bool
    evaluator: str
    reason: str


@dataclass(frozen=True)
class Provenance:
    task_id: str
    split: str
    model_hash: str
    source_hash: str
    seed: int


@dataclass(frozen=True)
class Experience:
    transition: ActualTransition
    evaluation: Evaluation
    provenance: Provenance


@dataclass(frozen=True)
class Rule:
    concept_id: str
    hypothesis: str
    confidence: float
    evidence_ids: tuple[str, ...]

    def __post_init__(self):
        if not 0 <= self.confidence <= 1 or not self.evidence_ids:
            raise ValueError("Rules require bounded confidence and evidence references")


def encode(record: Any) -> str:
    return canonical(
        {"schema": "mindscape.v3/1", "type": type(record).__name__, "data": asdict(record)}
    )


def _construct(cls, data):
    # Closed type registry and resolved annotations; never imports caller-specified types.
    from typing import get_args, get_origin, get_type_hints

    hints = get_type_hints(cls)
    if set(data) != set(hints):
        raise ValueError("Record fields do not match schema")

    def convert(t, value):
        if get_origin(t) is tuple:
            args = get_args(t)
            if args[-1:] == (Ellipsis,):
                return tuple(convert(args[0], x) for x in value)
            if len(value) != len(args):
                raise ValueError("Tuple length mismatch")
            return tuple(convert(a, b) for a, b in zip(args, value))
        if hasattr(t, "__dataclass_fields__"):
            return _construct(t, value)
        if t is float:
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("Finite number required")
            return float(value)
        if type(value) is not t:
            raise ValueError("Invalid field type")
        return value

    return cls(**{k: convert(hints[k], v) for k, v in data.items()})


def decode(text: str):
    if len(text) > 2_000_000:
        raise ValueError("Record too large")
    value = json.loads(text)
    registry = {
        c.__name__: c
        for c in (
            Entity,
            Relationship,
            Concept,
            Goal,
            Observation,
            State,
            Action,
            Prediction,
            ActualTransition,
            Evaluation,
            Provenance,
            Experience,
            Rule,
        )
    }
    if set(value) != {"schema", "type", "data"} or value["schema"] != "mindscape.v3/1":
        raise ValueError("Unsupported schema")
    if value["type"] not in registry:
        raise ValueError("Unknown record type")
    return _construct(registry[value["type"]], value["data"])
