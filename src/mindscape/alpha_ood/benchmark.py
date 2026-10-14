"""Fresh public arithmetic compositions; sealed labels belong to a custodian.

Importing this module performs no generation, prediction, fitting or scoring.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable
from dataclasses import dataclass

FAMILIES = ("operand_combinations", "long_sequences", "carry_chains", "borrow_chains",
            "operation_compositions", "transition_combinations")
OPS = ("add", "sub", "mul")


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Task:
    start: int
    steps: tuple[tuple[str, int], ...]
    split: str
    family: str

    def __post_init__(self) -> None:
        if type(self.start) is not int or self.start < 0:
            raise ValueError("nonnegative integer start required")
        if self.split not in ("train", "validation", "ood"):
            raise ValueError("unknown split")
        if not self.steps or len(self.steps) > 8:
            raise ValueError("one to eight transitions required")
        if self.family not in (*FAMILIES, "familiar"):
            raise ValueError("unknown family")
        if any(op not in OPS or type(n) is not int or n < 0 for op, n in self.steps):
            raise ValueError("invalid operation/operand")

    @property
    def identity(self) -> str:
        # Split/family deliberately excluded: relabeling cannot hide leakage.
        return digest({"start": self.start, "steps": self.steps})

    def historical_identities(self) -> set[str]:
        # Compare opaque historical V3/R5.2 IDs without opening their lockboxes.
        operands = [self.start, *[n for _, n in self.steps]]
        operations = [op for op, _ in self.steps]
        if len(operations) == 1 and operations[0] in ("add", "mul"):
            operands.sort()
        body = {"operands": operands, "operations": operations}
        return {self.identity, digest(body), digest({**body, "namespace": "R5.2-math"})}

    def public(self) -> dict:
        return {"identity": self.identity, "start": self.start, "steps": self.steps,
                "split": self.split, "family": self.family}


def reference(task: Task) -> tuple[int, ...]:
    """Custodian-only for sealed tasks; safe for training and synthetic fixtures."""
    value = task.start
    states = []
    for op, operand in task.steps:
        if op == "add":
            value += operand
        elif op == "sub":
            value -= operand
        else:
            value *= operand
        states.append(value)
    return tuple(states)


def make_public_suite(seed: int = 520054, train_n: int = 768,
                      validation_n: int = 96, family_n: int = 36) -> tuple[Task, ...]:
    """Generate inputs only. No OOD reference evaluation or label materialization.

    Distinct number ranges provide identity separation; a later custodian screen
    must certify observable context familiarity and historical identity exclusion.
    """
    rng = random.Random(seed)
    tasks: list[Task] = []
    seen: set[str] = set()

    def add(split: str, family: str, count: int) -> None:
        attempts = 0
        while sum(t.split == split and t.family == family for t in tasks) < count:
            attempts += 1
            if attempts > count * 1000:
                raise ValueError("generation exhausted")
            start = rng.randint(1, 99)
            length = rng.choice((2, 3))
            pattern = rng.choice((("add", "add"), ("sub", "sub"), ("mul", "mul")))
            steps = tuple((pattern[i % 2], rng.randint(1, 99)) for i in range(length))
            if family == "long_sequences":
                steps = tuple((rng.choice(OPS), rng.randint(1, 99))
                              for _ in range(rng.randint(5, 8)))
            elif family == "carry_chains":
                start = 10 ** rng.randint(4, 8) - 1
                steps = (("add", 1), ("add", rng.randint(1, 9)))
            elif family == "borrow_chains":
                start = 10 ** rng.randint(4, 8)
                steps = (("sub", 1), ("sub", rng.randint(1, 9)))
            elif family == "operation_compositions":
                steps = tuple((op, rng.randint(1, 99)) for op in
                              rng.choice((("add", "mul"), ("mul", "sub"),
                                          ("sub", "add"), ("mul", "add"))))
            elif family == "transition_combinations":
                # Familiar single-digit constituents, novel alternating schedule.
                start = rng.randint(1, 9)
                steps = (("add", rng.randint(1, 9)), ("sub", rng.randint(1, 9)),
                         ("add", rng.randint(1, 9)), ("sub", rng.randint(1, 9)))
            task = Task(start, steps, split, family)
            if task.identity not in seen:
                tasks.append(task)
                seen.add(task.identity)
    if min(train_n, validation_n, family_n) < 1:
        raise ValueError("positive counts required")
    # Chain construction has 45 identities per operation; frozen protocol uses 36.
    if family_n > 45:
        raise ValueError("chain family supports at most 45 unique identities")
    add("train", "familiar", train_n)
    add("validation", "familiar", validation_n)
    for family in FAMILIES:
        add("ood", family, family_n)
    return tuple(tasks)


def check_identities(tasks: Iterable[Task], historical_ids: set[str]) -> dict:
    tasks = tuple(tasks)
    ids = [t.identity for t in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate task identity, including across splits/families")
    if any(t.historical_identities() & historical_ids for t in tasks):
        raise ValueError("historical task overlap")
    return {"count": len(ids), "identity_hash": digest(sorted(ids)),
            "historical_registry_hash": digest(sorted(historical_ids))}


def memorization(task: Task, memory: dict[str, tuple[int, ...]]) -> tuple[int, ...]:
    """Exact training-task retrieval; misses emit zeros, never oracle repairs."""
    return memory.get(task.identity, (0,) * len(task.steps))


def transition_lookup(task: Task, table: dict[tuple[int, str, int], int]) -> tuple[int, ...]:
    """Training state/operation/operand lookup, executing predicted state feedback."""
    state = task.start
    states = []
    for op, operand in task.steps:
        state = table.get((state, op, operand), 0)
        states.append(state)
    return tuple(states)
