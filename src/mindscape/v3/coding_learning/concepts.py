"""Static evidence, never reference patches or inferred private outcomes."""

import ast
from collections import Counter
from dataclasses import asdict

from .schema import concept_record

FEATURES = (
    "Compare",
    "Gt",
    "GtE",
    "Lt",
    "LtE",
    "For",
    "While",
    "If",
    "Call",
    "Return",
    "Assign",
    "AugAssign",
    "ListComp",
    "Set",
    "Mult",
    "Add",
    "Sub",
    "ImportFrom",
    "mutation",
    "range",
    "sorted",
    "set",
    "max",
    "min",
    "sum",
    "len",
    "append",
    "contract_unchanged",
    "contract_order",
    "contract_boundary",
    "syntax_error",
    "files",
    "functions",
    "dependencies",
    "nodes",
)


def extract(repository, contract):
    counts = Counter()
    relationships = []
    functions = []
    mechanisms = set()
    for path, source in sorted(repository.items()):
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError):
            counts["syntax_error"] += 1
            continue
        counts["files"] += 1
        identifiers = {
            id(node): f"{path}:node{index}:{type(node).__name__}"
            for index, node in enumerate(ast.walk(tree))
        }
        for parent in ast.walk(tree):
            counts[type(parent).__name__] += 1
            counts["nodes"] += 1
            for child in ast.iter_child_nodes(parent):
                relationships.append(
                    (
                        identifiers[id(parent)],
                        "ast_child",
                        identifiers[id(child)],
                    )
                )
            if isinstance(parent, (ast.If, ast.For, ast.While)):
                for branch in ("body", "orelse"):
                    for statement in getattr(parent, branch):
                        relationships.append(
                            (
                                identifiers[id(parent)],
                                "controls_" + branch,
                                identifiers[id(statement)],
                            )
                        )
            if isinstance(parent, (ast.Assign, ast.AugAssign)):
                targets = parent.targets if isinstance(parent, ast.Assign) else [parent.target]
                stores = [
                    n.id for target in targets for n in ast.walk(target) if isinstance(n, ast.Name)
                ]
                loads = [n.id for n in ast.walk(parent.value) if isinstance(n, ast.Name)]
                for target in stores:
                    for source_name in loads:
                        relationships.append(
                            (path + ":" + target, "depends_on", path + ":" + source_name)
                        )
            if isinstance(parent, ast.FunctionDef):
                counts["functions"] += 1
                functions.append(path + ":" + parent.name)
                definitions = {a.arg for a in parent.args.args}
                for node in ast.walk(parent):
                    if isinstance(node, ast.Name):
                        relationships.append(
                            (
                                path + ":" + node.id,
                                "defines" if isinstance(node.ctx, ast.Store) else "uses",
                                path + ":" + parent.name,
                            )
                        )
                        if isinstance(node.ctx, ast.Store):
                            definitions.add(node.id)
                for name in sorted(definitions):
                    relationships.append((path + ":" + parent.name, "local_binding", name))
            if isinstance(parent, ast.ImportFrom):
                relationships.append((path, "imports", parent.module or ""))
                counts["dependencies"] += 1
            if isinstance(parent, ast.Call):
                name = (
                    parent.func.id
                    if isinstance(parent.func, ast.Name)
                    else (parent.func.attr if isinstance(parent.func, ast.Attribute) else "")
                )
                counts[name] += 1
                if name in ("sort", "reverse", "pop", "remove", "clear", "extend"):
                    counts["mutation"] += 1
                    mechanisms.add("possible_alias_mutation")
            if isinstance(parent, ast.Compare):
                mechanisms.add("boundary_predicate")
            if isinstance(parent, (ast.For, ast.While, ast.ListComp)):
                mechanisms.add("iteration_or_accumulator")
    text = (contract.description + " " + " ".join(contract.invariants)).lower()
    counts["contract_unchanged"] = int("unchanged" in text or "not mutate" in text)
    counts["contract_order"] = int("order" in text or "stable" in text)
    counts["contract_boundary"] = int(
        "inclusive" in text or "exclusive" in text or "at least" in text
    )
    if counts["set"] and counts["contract_order"]:
        mechanisms.add("possible_order_loss")
    if counts["ImportFrom"]:
        mechanisms.add("dependency_contract")
    return {
        "features": {k: counts[k] for k in FEATURES},
        "functions": functions,
        "relationships": sorted(set(relationships)),
        "concepts": [asdict(concept_record(m, sorted(functions))) for m in sorted(mechanisms)],
        "evidence_kind": "static_observation_not_test_result",
    }


def vector(view):
    import numpy as np

    return np.log1p([view["features"].get(k, 0) for k in FEATURES]).astype("float32")


def similarity(a, b):
    import numpy as np

    x, y = vector(a), vector(b)
    return float(np.dot(x, y) / max(1e-8, np.linalg.norm(x) * np.linalg.norm(y)))
