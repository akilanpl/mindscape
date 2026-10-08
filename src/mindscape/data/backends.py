"""Pluggable environment adapters; generic infrastructure contains no domain branches."""
from dataclasses import asdict
import hashlib
import json
from typing import Protocol

from mindscape.core.schema import Trajectory
from mindscape.core.serialization import decode
from mindscape.data.schemas import BenchmarkExample
from mindscape.environments.multiplication.environment import MultiplicationEnvironment
from mindscape.verification.verifier import verify

GENERATOR_VERSION = "multiplication-1.0.0"


class DatasetBackend(Protocol):
    name: str
    version: str
    def structures(self, spec): ...
    def structure_identity(self, category): ...
    def sample(self, category, rng, config): ...
    def identity(self, observation): ...
    def category(self, observation): ...
    def example(self, observation, seed, split): ...
    def validate(self, example): ...
    def assess(self, example, answer, trajectory): ...
    def reference(self, view): ...


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class MultiplicationBackend:
    name = "integer_multiplication"
    version = GENERATOR_VERSION

    def structures(self, spec):
        if "structures" in spec:
            values = spec["structures"]
        else:
            bounds = [spec[key] for key in ("min_digits_a", "max_digits_a", "min_digits_b", "max_digits_b")]
            if any(type(n) is not int or n < 1 for n in bounds) or bounds[0] > bounds[1] or bounds[2] > bounds[3]:
                raise ValueError("Invalid digit ranges")
            values = [f"{a}x{b}" for a in range(bounds[0], bounds[1] + 1)
                      for b in range(bounds[2], bounds[3] + 1)]
        for value in values:
            self.structure_identity(value)
        return list(values)

    def structure_identity(self, category):
        sizes = tuple(int(x) for x in category.split("x"))
        if len(sizes) != 2 or min(sizes) < 1:
            raise ValueError("Invalid digit structure")
        return tuple(sorted(sizes))

    def sample(self, category, rng, config):
        sizes = tuple(int(x) for x in category.split("x"))
        if len(sizes) != 2 or min(sizes) < 1:
            raise ValueError("Invalid digit structure")
        operands = []
        for size in sizes:
            low = 0 if size == 1 and config.get("include_zero", False) else 10 ** (size - 1)
            value = rng.randint(low, 10 ** size - 1)
            if config.get("signed", False) and rng.randrange(2):
                value = -value
            operands.append(value)
        return {"operands": operands, "source": "procedural_generator", "kind": "observation"}

    def identity(self, observation):
        operands = observation["operands"]
        if len(operands) != 2 or any(type(x) is not int for x in operands):
            raise ValueError("Invalid integer operand pair")
        return tuple(sorted(operands))

    def category(self, observation):
        return "x".join(str(len(str(abs(x)))) for x in observation["operands"])

    def example(self, observation, seed, split):
        from mindscape.core.schema import Observation
        env = MultiplicationEnvironment()
        env.reset(decode(Observation, observation))
        trajectory = env.run()
        ts = trajectory.transitions
        category = self.category(observation)
        return BenchmarkExample(
            stable_hash([self.name, self.identity(observation)]), self.name,
            " × ".join(map(str, observation["operands"])), observation,
            asdict(trajectory.initial_state), asdict(env.get_goal()), env.get_state().answer,
            [asdict(t) for t in ts],
            {"random_seed": seed, "operand_sizes": [int(x) for x in category.split("x")],
             "structural_category": category, "split": split, "generator_version": self.version,
             "difficulty": {"carry_count": sum(t.result.carry > 0 for t in ts),
                            "step_count": len(ts),
                            "multiplication_steps": sum(t.action.name == "multiply" for t in ts),
                            "max_operand_digits": max(len(str(abs(x))) for x in observation["operands"]),
                            "max_intermediate_value": max(abs(t.result.value) for t in ts)}})

    def validate(self, example):
        expected = self.example(example.observation, example.metadata["random_seed"],
                                example.metadata["split"])
        # Normalize tuples to JSON lists before comparison.
        if stable_hash(expected.to_dict()) != stable_hash(example.to_dict()):
            raise ValueError(f"Corrupt target/schema/metadata: {example.example_id}")

    def assess(self, example, answer, trajectory):
        if trajectory is None:
            return None, False, "unsupported_answer"
        try:
            typed = decode(Trajectory, trajectory)
            if stable_hash(asdict(typed.observation)) != stable_hash(example.observation):
                return False, False, "invalid_transition"
            outcome = verify(typed, answer)
        except (ValueError, TypeError, KeyError, AttributeError):
            return False, False, "malformed_trajectory"
        if not outcome.trajectory_valid:
            return False, False, "invalid_transition"
        if not outcome.goal_reached:
            return True, False, "goal_failure"
        return True, outcome.final_answer_valid, "correct" if outcome.final_answer_valid else "arithmetic_error"

    def reference(self, view):
        from mindscape.core.schema import Observation
        env = MultiplicationEnvironment()
        env.reset(decode(Observation, view["observation"]))
        trajectory = env.run()
        return env.get_state().answer, asdict(trajectory)


REGISTRY = {MultiplicationBackend.name: MultiplicationBackend()}


def register(backend):
    if backend.name in REGISTRY:
        raise ValueError("Backend already registered")
    REGISTRY[backend.name] = backend


BUILTIN_BACKENDS = {"integer_multiplication_claims": "mindscape.data.claims_backend"}


def get_backend(name):
    if name not in REGISTRY and name in BUILTIN_BACKENDS:
        import importlib
        importlib.import_module(BUILTIN_BACKENDS[name])
    try:
        return REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown environment: {name}") from exc

