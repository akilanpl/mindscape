"""Specification-preserving perturbations; hidden expected values remain private."""

import ast
import hashlib
import json
import re
from dataclasses import replace


def perturb(task, variant):
    repository = dict(task.repository)
    correct = dict(task.correct_repository)
    visible = json.loads(task.visible_tests)
    hidden = json.loads(task.hidden_tests)
    if variant in ("rename_functions", "rename_variables"):
        old = (
            visible["entry"].rsplit(".", 1)[1]
            if variant == "rename_functions"
            else next(iter(re.findall(r"items_\d+", "\n".join(repository.values()))), None)
        )
        if old:
            new = "renamed_function" if variant == "rename_functions" else "renamed_values"
            repository = {
                p: re.sub(r"\b" + re.escape(old) + r"\b", new, s) for p, s in repository.items()
            }
            correct = {
                p: re.sub(r"\b" + re.escape(old) + r"\b", new, s) for p, s in correct.items()
            }
            if variant == "rename_functions":
                visible["entry"] = visible["entry"].rsplit(".", 1)[0] + "." + new
                hidden["entry"] = visible["entry"]
    elif variant == "reordered_files":
        repository = dict(reversed(list(repository.items())))
        correct = dict(reversed(list(correct.items())))
    elif variant == "reordered_tests":
        visible["cases"].reverse()
        hidden["cases"].reverse()
    elif variant == "distractor_files":
        repository["irrelevant.py"] = "def unrelated(value):\n    return value + 99\n"
        correct["irrelevant.py"] = repository["irrelevant.py"]
    elif variant == "layout":

        def relocate(files):
            return {
                "package/" + p: re.sub(r"from (helper_\d+) import", r"from package.\1 import", s)
                for p, s in files.items()
            }

        repository = relocate(repository)
        correct = relocate(correct)
        visible["entry"] = "package." + visible["entry"]
        hidden["entry"] = visible["entry"]
    elif variant == "formatting":
        repository = {p: ast.unparse(ast.parse(s)) + "\n" for p, s in repository.items()}
        correct = {p: ast.unparse(ast.parse(s)) + "\n" for p, s in correct.items()}
    else:
        raise ValueError("Unknown perturbation")
    patch = {p: s for p, s in correct.items() if repository.get(p) != s}
    identity = hashlib.sha256(
        (task.task_id + variant + json.dumps(repository, sort_keys=True)).encode()
    ).hexdigest()
    return replace(
        task,
        task_id=identity,
        repository=repository,
        correct_repository=correct,
        ground_truth_patch=patch,
        visible_tests=json.dumps(visible),
        hidden_tests=json.dumps(hidden),
        metadata=task.metadata | {"perturbation": variant, "parent_id": task.task_id},
    )
