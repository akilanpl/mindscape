"""Independent generated repairs; learner JSON excludes private tests and teacher code."""

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

from mindscape.v3.schema import digest

from .schema import Contract, RepairTask, task_from_dict

TRAIN_FAMILIES = ("threshold", "exclusive", "mutation", "ordering", "accumulator", "empty")
OOD_FAMILIES = (
    "cross_file",
    "compound",
    "control_flow",
    "renaming",
    "alias_mutation",
    "composition",
)


def make_task(split, family, seed, k):
    rng = random.Random(seed)
    fn = "compute_" + str(rng.randrange(10000, 99999))
    entry = "solution." + fn
    invariant = ("Inputs remain unchanged.",)
    cases = []
    description = ""
    correct = ""
    broken = ""

    def case(args, expected):
        cases.append(
            {
                "args": args,
                "expected": expected,
                "expected_type": type(expected).__name__,
                "unchanged": True,
            }
        )

    if family in ("threshold", "renaming", "cross_file"):
        description = f"Return the sum of values at least {k}, inclusive. Do not mutate the input."
        arg = "measurements_zeta" if family == "renaming" else "values"
        correct = f"def {fn}({arg}):\n    return sum(item for item in {arg} if item >= {k})\n"
        broken = correct.replace(">=", ">")
        for xs in ([k], [k - 1, k, k + 2], [], [-k, 0, k], [k, k], [k + 3, k - 2]):
            case([xs], sum(x for x in xs if x >= k))
        if family == "cross_file":
            correct = {
                "rules.py": f"def eligible(value):\n    return value >= {k}\n",
                "solution.py": f"from rules import eligible\ndef {fn}(values):\n    return sum(x for x in values if eligible(x))\n",
            }
            broken = dict(correct)
            broken["rules.py"] = correct["rules.py"].replace(">=", ">")
    elif family in ("exclusive", "control_flow"):
        description = (
            f"Return the sum of integers from {k} inclusive to n exclusive; zero when n <= {k}."
        )
        correct = f"def {fn}(n):\n    return sum(range({k}, n))\n"
        broken = correct.replace("n))", "n+1))")
        if family == "control_flow":
            correct = f"def {fn}(n):\n    total = 0\n    cursor = {k}\n    while cursor < n:\n        total += cursor\n        cursor += 1\n    return total\n"
            broken = correct.replace("cursor < n", "cursor <= n")
        for n in (k, k + 1, k + 4, k - 1, k + 8, k - 4):
            case([n], sum(range(k, n)))
    elif family in ("mutation", "alias_mutation"):
        description = f"Return the first {k} items of the ascending sorted input. Inputs must remain unchanged."
        correct = f"def {fn}(values):\n    result = sorted(values)\n    return result[:{k}]\n"
        broken = f"def {fn}(values):\n    values.sort()\n    return values[:{k}]\n"
        if family == "alias_mutation":
            correct = f"def arrange(items):\n    local = list(items)\n    local.sort()\n    return local\ndef {fn}(values):\n    return arrange(values)[:{k}]\n"
            broken = correct.replace("local = list(items)", "local = items")
        for xs in ([3, 1, 2], [], [2, 2, 1], [-1, 4, 0], list(range(k + 2, 0, -1)), [0]):
            case([xs], sorted(xs)[:k])
    elif family in ("ordering", "composition"):
        description = f"Return distinct values at least {k}, in their original first-occurrence order; keep inputs unchanged."
        correct = f"def {fn}(values):\n    result = []\n    for value in values:\n        if value >= {k} and value not in result:\n            result.append(value)\n    return result\n"
        broken = f"def {fn}(values):\n    return sorted(set(x for x in values if x >= {k}))\n"
        if family == "composition":
            broken = broken.replace(">=", ">")
        for xs in (
            [k + 2, k, k + 2, k + 1],
            [k, k],
            [],
            [k - 1, k + 2, k + 1],
            [k + 3, k + 1, k + 3],
            [k - 2],
        ):
            case([xs], list(dict.fromkeys(x for x in xs if x >= k)))
    elif family in ("accumulator", "compound"):
        description = (
            f"Return the product of (value + {k}) for every input value; empty product is one."
            if family == "accumulator"
            else f"Return the product of input values at least {k}, inclusive; empty product is one."
        )
        expression = f"value + {k}" if family == "accumulator" else "value"
        guard = (
            "" if family == "accumulator" else f"        if value < {k}:\n            continue\n"
        )
        correct = f"def {fn}(values):\n    result = 1\n    for value in values:\n{guard}        result *= {expression}\n    return result\n"
        broken = correct.replace("result = 1", "result = 0")
        if family == "compound":
            broken = broken.replace(f"value < {k}", f"value <= {k}")
        for xs in ([k], [], [1, 2], [k - 1, k, k + 1], [0], [k + 2, k + 3]):
            answer = 1
            for x in xs:
                if family == "accumulator":
                    answer *= x + k
                elif x >= k:
                    answer *= x
            case([xs], answer)
    elif family == "empty":
        description = (
            f"Return the largest input value, or {k} for an empty input. Inputs remain unchanged."
        )
        correct = f"def {fn}(values):\n    return max(values, default={k})\n"
        broken = f"def {fn}(values):\n    return max(values)\n"
        for xs in ([], [k], [1, 3, 2], [-1, -2], [0], []):
            case([xs], max(xs, default=k))
    else:
        raise ValueError("Unknown family")
    if isinstance(correct, str):
        correct = {"solution.py": correct}
        broken = {"solution.py": broken}
    contract = Contract(
        entry,
        description,
        ("JSON-compatible integers or integer list",),
        cases[0]["expected_type"],
        invariant,
    )
    visible = {"entry": entry, "cases": cases[:2]}
    private = {"entry": entry, "cases": cases[2:]}
    ident = digest({"repository": broken, "contract": asdict(contract), "visible_tests": visible})
    task = RepairTask(ident, split, family, broken, contract, visible)
    return task, {"identity": ident, "private_tests": private, "teacher_repository": correct}


def freeze(destination, executor):
    root = Path(destination)
    if root.exists():
        raise ValueError("Frozen dataset destination must be new")
    rng = random.Random(840013)
    ks = iter(rng.sample(range(2, 180), 60))
    public = []
    private = []
    receipts = []
    for split, families, repeats in (
        ("train", TRAIN_FAMILIES, 4),
        ("validation", TRAIN_FAMILIES, 1),
        ("iid", TRAIN_FAMILIES, 1),
        ("ood", OOD_FAMILIES, 2),
    ):
        for family in families:
            for repeat in range(repeats):
                task, secret = make_task(split, family, rng.randrange(10**9), next(ks))
                outcomes = {}
                for label, repo in (
                    ("broken", task.repository),
                    ("teacher", secret["teacher_repository"]),
                ):
                    for name, suite in (
                        ("visible", task.visible_tests),
                        ("private", secret["private_tests"]),
                    ):
                        outcomes[label + "_" + name] = executor.run(repo, suite)
                if (
                    outcomes["broken_visible"]["all_passed"]
                    or outcomes["broken_private"]["all_passed"]
                    or not outcomes["teacher_visible"]["all_passed"]
                    or not outcomes["teacher_private"]["all_passed"]
                ):
                    raise ValueError(
                        "Generated defect/oracle failed actual execution validation: "
                        + task.identity
                    )
                public.append(asdict(task))
                private.append(secret)
                receipts.append({"task_id": task.identity, "executions": outcomes})
    ids = [t["identity"] for t in public]
    if len(ids) != len(set(ids)):
        raise ValueError("Task identity leakage")
    root.mkdir(parents=True)
    pub = {"schema": "mindscape.v3.coding/1", "tasks": public}
    sec = {"schema": "mindscape.v3.coding.scorer/1", "tasks": private}
    manifest = {
        "public_hash": digest(pub),
        "private_hash": digest(sec),
        "task_ids": ids,
        "counts": {
            s: sum(t["split"] == s for t in public) for s in ("train", "validation", "iid", "ood")
        },
        "validation_hash": digest(receipts),
        "seed": 840013,
        "reference_policy": "teacher train-only; scorer terminal-only",
    }
    for name, value in (
        ("public.json", pub),
        ("private.json", sec),
        ("manifest.json", manifest),
        ("validation.json", receipts),
    ):
        (root / name).write_text(json.dumps(value, indent=2) + "\n")
    return manifest


def load_public(root):
    root = Path(root)
    data = json.loads((root / "public.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    if digest(data) != manifest["public_hash"]:
        raise ValueError("Public manifest mismatch")
    tasks = [task_from_dict(row) for row in data["tasks"]]
    if len({t.identity for t in tasks}) != len(tasks):
        raise ValueError("Identity overlap")
    for t in tasks:
        v = t.visible()
        v.pop("task_id")
        if digest(v) != t.identity:
            raise ValueError("Task identity changed")
    return tasks, manifest


def load_scorer(root):
    root = Path(root)
    sec = json.loads((root / "private.json").read_text())
    if digest(sec) != json.loads((root / "manifest.json").read_text())["private_hash"]:
        raise ValueError("Private manifest mismatch")
    return {row["identity"]: row for row in sec["tasks"]}


if __name__ == "__main__":
    from .execution import Executor

    p = argparse.ArgumentParser()
    p.add_argument("--destination", required=True)
    p.add_argument("--runtime", required=True)
    a = p.parse_args()
    print(json.dumps(freeze(a.destination, Executor(a.runtime))))
