import copy
import json
from pathlib import Path

import pytest

from mindscape.r5_2_math.arithmetic import run
from mindscape.r5_2_math.capability import Request, solve
from mindscape.r5_2_math.data import build
from mindscape.r5_2_math.verification import Teacher, verify


@pytest.mark.parametrize(
    "a,b,op",
    [
        (99999, 1, "add"),
        (100000, 1, "sub"),
        (-9876, 321, "mul"),
        (-543, -98, "sub"),
        (0, -77, "mul"),
    ],
)
def test_reference_execution_and_signed_variable_lengths(a, b, op):
    task = {"operands": [a, b], "operations": [op]}
    result = run(task, Teacher())
    assert verify(task, result)["strict"]


def test_bad_neural_claim_never_fixed_by_verifier():
    class Wrong(Teacher):
        def local(self, state):
            return 0, 0, {"kind": "test_bad_claim"}

    task = {"operands": [19, 8], "operations": ["add"]}
    result = run(task, Wrong())
    before = copy.deepcopy(result)
    score = verify(task, result)
    assert not score["exact"] and result == before and result["answer"] == "0"


@pytest.mark.parametrize(
    "family,inputs",
    [
        ("division", {"a": -17, "b": 5}),
        ("fractions", {"a": [1, 3], "b": [2, 5], "operation": "add"}),
        ("algebra", {"coefficients": [1, 2, 3], "x": 4}),
        ("equation", {"a": 2, "b": 3, "c": 11}),
        ("matrix", {"a": [[1, 2], [3, 4]], "b": [[1], [2]]}),
        ("geometry", {"a": [0, 0], "b": [3, 4]}),
        ("calculus", {"coefficients": [0, 1, 2], "lower": 0, "upper": 2}),
        ("statistics", {"values": [1, 2, 3, 4]}),
        ("probability", {"successes": 2, "trials": 7}),
    ],
)
def test_modular_symbolic_families_are_explicitly_not_neural(family, inputs):
    response = solve(Request(family, inputs))
    assert response["method"] == "symbolic_tool" and response["verification"]["success"]


def test_no_symbolic_fallback_or_private_input():
    with pytest.raises(ValueError):
        solve(Request("integer", {"operands": [1, 2], "operations": ["add"]}))
    with pytest.raises(ValueError):
        Request("integer", {"operands": [1, 2], "operations": ["add"], "expected": 3})
    with pytest.raises(ValueError):
        solve(Request("division", {"a": 1, "b": 0}))


def test_dataset_determinism_and_disjoint_frozen_identities():
    p = json.loads(Path("configs/r5_2_math/protocol.json").read_text())
    m = build(p)
    assert m == build(p) and len(m["tasks"]) == 972
    assert len({t["identity"] for t in m["tasks"]}) == 972
    assert sum(t["split"] == "ood" for t in m["tasks"]) == 144


def test_held_local_labels_removed_from_memory_and_loss():
    from mindscape.r5_2_math.training import train_examples

    task = {"split": "train", "operands": [8, 7], "operations": ["add"]}
    from mindscape.r5_2_math.verification import demonstration

    seq, _signs, memory, _sign_memory, exposure = train_examples(
        [task], [demonstration(task)], {("add", 8, 7, 0)}
    )
    assert exposure == 1 and "add|8|7|0" not in memory and seq[0][0][3] == 0


def test_neural_source_has_no_reference_answer_or_arithmetic_product_input():
    import ast

    tree = ast.parse(Path("src/mindscape/r5_2_math/model.py").read_text())
    imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not any("verification" in str(n) for n in imports)
    assert not any(isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult) for n in ast.walk(tree))


def test_euclidean_division_and_invalid_fraction_boundary():
    response = solve(Request("division", {"a": -17, "b": -5}))
    assert response["result"] == {"quotient": "4", "remainder": "3"}
    with pytest.raises(ValueError):
        solve(Request("fractions", {"a": [1, 0], "b": [2, 3], "operation": "add"}))
    with pytest.raises(ValueError):
        Request("equation", {"a": '__import__("os")', "b": 0, "c": 1})
