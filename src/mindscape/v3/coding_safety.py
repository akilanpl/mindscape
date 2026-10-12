"""Versioned hardening outside immutable coding-study source.

Use GuardedExecutor for new experiments. Frozen study drivers retain their
original executor and are independently replayed under this stricter policy.
WASI remains the isolation boundary; this is not a general Python security proof.
"""

import ast

from .coding_learning.execution import Executor, safe_source

POLICY = "mindscape.v3.coding.safety/1"


def guarded_source(repository):
    safe_source(repository)
    for source in repository.values():
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Name) and node.id.startswith("__"):
                raise ValueError("Indirect reflective identifier forbidden")
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ) and node.name.startswith("__"):
                raise ValueError("Reflective definition forbidden")
    return True


def same_types(actual, expected):
    """JSON-domain recursive type agreement, including bool versus int."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            same_types(a, b) for a, b in zip(actual, expected, strict=True)
        )
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            same_types(actual[k], expected[k]) for k in expected
        )
    return True


class GuardedExecutor(Executor):
    def run(self, repository, suite):
        try:
            guarded_source(repository)
        except (SyntaxError, ValueError) as error:
            self.calls += 1
            return {
                "all_passed": False,
                "passed": 0,
                "total": len(suite["cases"]),
                "case_passes": [False] * len(suite["cases"]),
                "error": str(error),
                "execution": None,
                "observed": [],
                "actual_vm_call": False,
                "guard_policy": POLICY,
            }
        result = super().run(repository, suite)
        passed = []
        for i, case in enumerate(suite["cases"]):
            observed = result["observed"][i] if i < len(result["observed"]) else {}
            okay = result["case_passes"][i] and same_types(observed.get("value"), case["expected"])
            if case.get("unchanged"):
                okay = okay and same_types(observed.get("args_after"), case["args"])
            passed.append(bool(okay))
        result.update(
            case_passes=passed,
            passed=sum(passed),
            all_passed=bool(passed) and all(passed),
            guard_policy=POLICY,
        )
        return result
