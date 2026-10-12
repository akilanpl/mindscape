"""Small bounded domains. Candidate sets depend on syntax, never semantic correctness.

Coding is an AST-interpreted Python subset, NOT arbitrary Python execution. Use the
preserved WASI sandbox for a future general Python adapter, never host exec/eval.
"""

import ast
from dataclasses import dataclass
from typing import Protocol

from .schema import Action, ActualTransition, Evaluation, Goal, Observation, State, digest


@dataclass(frozen=True)
class Task:
    domain: str
    inputs: tuple[int, ...]
    split: str
    family: str

    def __post_init__(self):
        sizes = {"mathematics": 2, "python": 3, "dynamics": 3}
        if self.domain not in sizes or len(self.inputs) != sizes[self.domain]:
            raise ValueError("Unknown domain or wrong input shape")
        if type(self.inputs) is not tuple or any(
            type(x) is not int or abs(x) > 32 for x in self.inputs
        ):
            raise ValueError("Bounded integer inputs required")
        if self.split not in ("train", "validation", "iid", "ood"):
            raise ValueError("Unknown task partition")

    @property
    def identity(self):
        inputs = tuple(sorted(self.inputs)) if self.domain == "mathematics" else self.inputs
        return digest({"domain": self.domain, "inputs": inputs})


class Domain(Protocol):
    name: str

    def observe(self, task: Task) -> Observation: ...
    def candidates(self, state: State) -> tuple[Action, ...]: ...
    def execute(self, state: State, action: Action) -> ActualTransition: ...
    def evaluate(self, task: Task, transition: ActualTransition) -> Evaluation: ...


class BoundedDomain:
    name = ""
    instruction = ""

    def observe(self, task):
        if task.domain != self.name:
            raise ValueError("Wrong domain")
        return Observation(self.name, tuple(float(x) for x in task.inputs), Goal(self.instruction))

    def candidates(self, state):
        # Same 65 candidates for every task, including wrong answers. No valid_actions oracle.
        return tuple(Action("submit", x) for x in range(-32, 33))

    def execute(self, state, action):
        if state.terminal or action not in self.candidates(state):
            raise ValueError("Invalid syntax or terminal state")
        after = State(state.observation, (float(action.value),), True)
        return ActualTransition(state, action, after, "proposal_written_unchanged")

    def expected(self, inputs):
        raise NotImplementedError

    def evaluate(self, task, transition):
        # Private post-action evaluation: never a model input or candidate generator.
        binding = transition.before.observation == self.observe(task)
        valid = (
            binding
            and transition.action in self.candidates(transition.before)
            and transition == self.execute(transition.before, transition.action)
        )
        success = valid and transition.after.written == (float(self.expected(task.inputs)),)
        return Evaluation(
            valid,
            success,
            f"{self.name}.independent.v3",
            "matched" if success else "incorrect_or_unbound",
        )


class Mathematics(BoundedDomain):
    name = "mathematics"
    instruction = "Submit the sum of the two observed integers."

    def expected(self, inputs):
        return sum(inputs)


class Dynamics(BoundedDomain):
    name = "dynamics"
    instruction = "Submit position after one unit time: observed position, velocity, acceleration."

    def expected(self, inputs):
        x, v, a = inputs
        # Discrete semi-implicit Euler dynamics, explicit in task semantics.
        return x + v + a


def interpret_python(source: str, x: int, limit: int = 64) -> int:
    """Evaluate a tiny pure Python expression subset with bounded integer arithmetic."""
    if len(source) > 512 or type(x) is not int or abs(x) > 32:
        raise ValueError("Resource bound exceeded")
    tree = ast.parse(source, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > limit:
        raise ValueError("AST budget exceeded")

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Name) and node.id == "x":
            return x
        if isinstance(node, ast.Constant) and type(node.value) is int and abs(node.value) <= 32:
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            result = -visit(node.operand)
        elif isinstance(node, ast.BinOp) and type(node.op) in (ast.Add, ast.Sub, ast.Mult):
            left, right = visit(node.left), visit(node.right)
            result = {
                ast.Add: lambda: left + right,
                ast.Sub: lambda: left - right,
                ast.Mult: lambda: left * right,
            }[type(node.op)]()
        else:
            raise ValueError("Python construct outside capability subset")
        if abs(result) > 1024:
            raise ValueError("Integer budget exceeded")
        return result

    return visit(tree)


class PythonCoding(BoundedDomain):
    name = "python"
    instruction = (
        "Repair offset in Python expression x * coefficient + offset from input/output example."
    )

    def observe(self, task):
        x, coefficient, offset = task.inputs
        return Observation(
            self.name,
            (float(x), float(coefficient), float(x * coefficient + offset)),
            Goal(self.instruction),
        )

    def candidates(self, state):
        return tuple(Action("patch_offset", x) for x in range(-32, 33))

    def execute(self, state, action):
        if state.terminal or action not in self.candidates(state):
            raise ValueError("Invalid patch")
        coefficient = int(state.observation.values[1])
        source = f"x * {coefficient} + {action.value}"
        outputs = tuple(float(interpret_python(source, x)) for x in (-2, 0, 2))
        return ActualTransition(
            state, action, State(state.observation, outputs, True), "bounded_python_patch_executed"
        )

    def evaluate(self, task, transition):
        valid = transition.before == State(self.observe(task)) and transition == self.execute(
            transition.before, transition.action
        )
        _, coefficient, offset = task.inputs
        expected = tuple(float(x * coefficient + offset) for x in (-2, 0, 2))
        success = valid and transition.after.written == expected
        return Evaluation(
            valid,
            success,
            "python.independent.v3",
            "hidden_subset_tests_passed" if success else "patch_failed",
        )


DOMAINS = {d.name: d for d in (Mathematics(), PythonCoding(), Dynamics())}


def generate_tasks():
    tasks = []
    # Prospectively frozen input ranges; validation and tests have distinct identities.
    ranges = {
        "train": range(-3, 1),
        "validation": range(1, 2),
        "iid": range(2, 4),
        "ood": range(5, 7),
    }
    for domain in DOMAINS:
        for split, values in ranges.items():
            for a in values:
                for b in range(7, 12) if domain == "mathematics" else range(-2, 3):
                    inputs = (
                        (a, b)
                        if domain == "mathematics"
                        else (a, b, a + b)
                        if domain == "python"
                        else (a, b, 1)
                    )
                    tasks.append(
                        Task(
                            domain,
                            inputs,
                            split,
                            "extrapolation" if split == "ood" else "bounded_integer",
                        )
                    )
    return tuple(tasks)
