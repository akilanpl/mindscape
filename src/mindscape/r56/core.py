"""Bounded typed transitions; deterministic evidence is not learned reasoning."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Callable, Mapping

STATUS = "EXPERIMENTAL — GENERALIZATION NOT YET DEMONSTRATED"
Scalar = int | float | str | bool


@dataclass(frozen=True)
class State:
    domain: str
    values: tuple[tuple[str, Scalar], ...]

    @classmethod
    def create(cls, domain: str, **values: Scalar) -> State:
        if not domain or any(type(v) not in (int, float, str, bool) for v in values.values()):
            raise ValueError("A state requires a domain and scalar typed values")
        return cls(domain, tuple(sorted(values.items())))

    def mapping(self) -> dict[str, Scalar]:
        return dict(self.values)

    @property
    def identity(self) -> str:
        return hashlib.sha256(json.dumps([self.domain, self.values], allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Action:
    concept: str
    parameters: tuple[tuple[str, Scalar], ...] = ()

    @classmethod
    def create(cls, concept: str, **parameters: Scalar) -> Action:
        return cls(concept, tuple(sorted(parameters.items())))


@dataclass(frozen=True)
class Concept:
    name: str
    domain: str
    fields: Mapping[str, type]
    parameter_fields: Mapping[str, type]
    precondition: Callable[[dict, dict], bool]
    transition: Callable[[dict, dict], State]
    postcondition: Callable[[dict, dict, dict], bool]
    abstract_properties: tuple[str, ...] = ()

    def validate(self, state: State, action: Action) -> None:
        values, parameters = state.mapping(), dict(action.parameters)
        if state.domain != self.domain or action.concept != self.name:
            raise ValueError("Concept/domain mismatch")
        for actual, schema in ((values, self.fields), (parameters, self.parameter_fields)):
            if set(actual) != set(schema) or any(type(actual[k]) is not t for k, t in schema.items()):
                raise ValueError("State/action does not match exact typed schema")
        if not self.precondition(values, parameters):
            raise ValueError("Transition precondition failed")


class Registry:
    def __init__(self) -> None:
        self._concepts: dict[str, Concept] = {}

    def register(self, concept: Concept) -> None:
        if concept.name in self._concepts:
            raise ValueError("Duplicate concept")
        self._concepts[concept.name] = concept

    def get(self, name: str) -> Concept:
        if name not in self._concepts:
            raise ValueError(f"Unknown concept: {name}")
        return self._concepts[name]

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._concepts))


@dataclass(frozen=True)
class Evidence:
    concept: str
    before: State
    proposed: State
    verified: bool
    source: str
    verifier: str = "independent-domain-postcondition"

    def as_dict(self) -> dict:
        return {"concept": self.concept, "before": self.before.mapping(),
                "proposed": self.proposed.mapping(), "domain": self.before.domain,
                "before_id": self.before.identity, "after_id": self.proposed.identity,
                "verified": self.verified, "source": self.source, "verifier": self.verifier}


class Composer:
    def __init__(self, registry: Registry):
        self.registry = registry

    def verify(self, before: State, action: Action, proposed: State,
               source: str = "deterministic-rule") -> Evidence:
        concept = self.registry.get(action.concept)
        concept.validate(before, action)
        # Never repair a prediction before recording/checking it.
        valid_schema = proposed.domain == concept.domain and set(proposed.mapping()) == set(concept.fields)
        valid_types = valid_schema and all(type(proposed.mapping()[k]) is t for k, t in concept.fields.items())
        passed = bool(valid_types and concept.postcondition(before.mapping(), dict(action.parameters), proposed.mapping()))
        return Evidence(concept.name, before, proposed, passed, source)

    def run(self, initial: State, actions: list[Action], max_steps: int = 100) -> tuple[State, tuple[Evidence, ...]]:
        if len(actions) > max_steps or max_steps < 0:
            raise ValueError("Composition exceeds step budget")
        current, records = initial, []
        for action in actions:
            concept = self.registry.get(action.concept)
            concept.validate(current, action)
            proposed = concept.transition(current.mapping(), dict(action.parameters))
            evidence = self.verify(current, action, proposed)
            records.append(evidence)
            if not evidence.verified:
                raise ValueError("Transition failed independent verification")
            current = proposed
        return current, tuple(records)


def encode_state(state: State, fields: tuple[str, ...]) -> tuple[float, ...]:
    """Explicit field order; strings require a domain-specific encoder."""
    values = state.mapping()
    if set(values) != set(fields) or any(type(values[k]) not in (int, float) for k in fields):
        raise ValueError("Numeric encoder requires an exact numeric field schema")
    return tuple(float(values[k]) for k in fields)
