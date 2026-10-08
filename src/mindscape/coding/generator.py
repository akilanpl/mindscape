"""Parameterized repositories with private independent input/output cases."""

import hashlib
import json
import random

from mindscape.coding.schema import CodeTask

TRAIN = [
    "conditional",
    "off_by_one",
    "operator",
    "return_value",
    "initialization",
    "loop",
    "missing_branch",
    "list_handling",
    "type_mismatch",
    "exception_handling",
]
OOD = ["function_interaction", "cross_file", "compound", "unfamiliar_structure"]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def task(seed, structure, split="train"):
    r = random.Random(seed)
    name = "repair_" + str(seed)
    x = "items_" + str(seed % 997)
    k = r.randint(2, 15)
    module = "module_" + str(seed % 101)
    path = f"{module}.py"
    entry = f"{module}.{name}"
    if structure == "conditional":
        good = f"def {name}(n):\n    return n >= {k}\n"
        bad = good.replace(">=", ">")
        inputs = [k - 1, k, k + 1, k + 9]
        outputs = [False, True, True, True]
        statement = f"Return whether n is at least {k}."
    elif structure == "off_by_one":
        good = f"def {name}(n):\n    return sum(range(n + 1))\n"
        bad = good.replace("n + 1", "n")
        inputs = [0, 1, k, k + 1]
        outputs = [n * (n + 1) // 2 for n in inputs]
        statement = "Return the sum of integers from zero through n, inclusive."
    elif structure == "operator":
        good = f"def {name}(n):\n    return n * {k}\n"
        bad = good.replace(" * ", " + ")
        inputs = [0, 1, k, -3]
        outputs = [n * k for n in inputs]
        statement = f"Return n multiplied by {k}."
    elif structure == "return_value":
        good = f"def {name}(n):\n    if n < {k}:\n        return -1\n    return 1\n"
        bad = good.replace("return -1", "return 0")
        inputs = [0, k, k - 1, k + 1]
        outputs = [-1, 1, -1, 1]
        statement = f"Return -1 below {k}, otherwise 1."
    elif structure == "initialization":
        good = f"def {name}({x}):\n    total = {k}\n    for value in {x}:\n        total += value\n    return total\n"
        bad = good.replace(f"total = {k}", "total = 0")
        inputs = [[], [1], [2, 3], [-k, k]]
        outputs = [k + sum(v) for v in inputs]
        statement = f"Sum the input list and add an initial offset of {k}."
    elif structure == "loop":
        good = f"def {name}({x}):\n    return sum(value for value in {x})\n"
        bad = good.replace(f"in {x})", f"in {x}[:-1])")
        inputs = [[], [1], [1, 2, 3], [k, -1]]
        outputs = [sum(v) for v in inputs]
        statement = "Return the sum of every input list element."
    elif structure == "missing_branch":
        good = f"def {name}(n):\n    if n < 0:\n        return -1\n    if n > 0:\n        return 1\n    return 0\n"
        bad = good.replace("return -1", "return 0")
        inputs = [0, 1, -1, -k]
        outputs = [0, 1, -1, -1]
        statement = "Return the sign: -1 for negative, 0 for zero, 1 for positive."
    elif structure == "list_handling":
        good = f"def {name}({x}):\n    return list(reversed({x}))\n"
        bad = good.replace(f"list(reversed({x}))", f"list({x})")
        inputs = [[], [1], [1, 2], [k, 2, 3]]
        outputs = [list(reversed(v)) for v in inputs]
        statement = "Return a new list with the input elements in reverse order."
    elif structure == "type_mismatch":
        good = f"def {name}({x}):\n    return list({x})\n"
        bad = good.replace(f"list({x})", f"tuple({x})")
        inputs = [[], [1], [1, 2], [k, 3]]
        outputs = inputs
        statement = "Return a list copy of the input; the output type must be list."
    elif structure == "exception_handling":
        good = f"def {name}(n):\n    try:\n        return {k} / n\n    except ZeroDivisionError:\n        return 0\n"
        bad = good.replace("ZeroDivisionError", "TypeError")
        inputs = [1, k, 0, -k]
        outputs = [float(k), 1.0, 0, -1.0]
        statement = f"Return {k}/n, and return 0 when n is zero."
    elif structure in ("function_interaction", "cross_file"):
        helper = f"def adjust(n):\n    return n + {k}\n"
        good = (
            helper
            if structure == "function_interaction"
            else f"from helper_{seed % 97} import adjust\n"
        ) + f"def {name}(n):\n    return adjust(n) * 2\n"
        bad = good.replace("adjust(n) * 2", "adjust(n * 2)")
        inputs = [0, 1, -k, k]
        outputs = [2 * (n + k) for n in inputs]
        statement = f"Adjust n upward by {k}, then double the adjusted value."
    elif structure == "compound":
        good = f"def {name}(n):\n    return sum(range(n + 1)) * {k}\n"
        bad = good.replace("n + 1", "n").replace(" * ", " + ")
        inputs = [0, 1, k, k + 2]
        outputs = [n * (n + 1) // 2 * k for n in inputs]
        statement = f"Sum zero through n inclusive, then multiply the sum by {k}."
    else:
        good = f"def {name}({x}):\n    return [value for value in {x} if value % 2 == 0]\n"
        bad = good.replace("== 0", "== 1")
        inputs = [[], [1, 2], [2, 4, 5], [k, k + 1]]
        outputs = [[v for v in values if v % 2 == 0] for values in inputs]
        statement = "Return only even numbers, preserving input order."
    repository = {path: bad}
    correct = {path: good}
    if structure == "cross_file":
        repository[f"helper_{seed % 97}.py"] = helper
        correct[f"helper_{seed % 97}.py"] = helper
    cases = [
        {"args": [v], "expected": o, "expected_type": type(o).__name__}
        for v, o in zip(inputs, outputs)
    ]
    visible = cases[:2]
    hidden = cases
    tests = json.dumps({"entry": entry, "cases": visible}, sort_keys=True)
    private = json.dumps({"entry": entry, "cases": hidden}, sort_keys=True)
    metadata = {
        "split": split,
        "structural_category": structure,
        "bug_category": structure,
        "difficulty": "compound" if structure in OOD else "local",
        "files_touched": 1,
        "function_count": good.count("def ") + (1 if structure == "cross_file" else 0),
        "test_count": 4,
        "dependency_count": 0,
        "expected_action_count": 5 if structure != "compound" else 7,
        "seed": seed,
        "generator_version": "coding-1.0",
        "entry": entry,
        "failure_signature": "private execution validation required",
    }
    return CodeTask(
        digest([seed, structure, repository]),
        repository,
        statement,
        tests,
        private,
        correct,
        {path: good},
        metadata,
    )


def generate(seed=7001, counts=None):
    counts = counts or {"train": 1000, "validation": 20, "test": 40, "ood_test": 40}
    rng = random.Random(seed)
    result = {}
    seen = set()
    for split, count in counts.items():
        result[split] = []
        for i in range(count):
            structure = (OOD if split == "ood_test" else TRAIN)[
                i % len(OOD if split == "ood_test" else TRAIN)
            ]
            t = task(rng.randrange(100000, 999999999), structure, split)
            assert t.task_id not in seen
            seen.add(t.task_id)
            result[split].append(t)
    return result
