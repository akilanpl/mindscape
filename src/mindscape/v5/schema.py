"""Public graph contract: entities, directed typed relations, and proposed actions only."""

import math

DIMS = {"mathematics": 6, "coding": 10, "physics": 8}
OUTPUTS = {"mathematics": 1, "coding": 1, "physics": 4}
SCALES = {"mathematics": 100.0, "coding": 100.0, "physics": 4.0}
MAX_NODES = 7


def validate(graph):
    if set(graph) != {"domain", "entities", "observed", "relations", "action"}:
        raise ValueError("Unknown or private graph fields")
    domain = graph["domain"]
    if domain not in DIMS:
        raise ValueError("Unknown domain")
    entities = graph["entities"]
    observed = graph["observed"]
    relations = graph["relations"]
    n = len(entities)
    if not 1 <= n <= MAX_NODES or len(observed) != n or len(relations) != n:
        raise ValueError("Entity budget")
    if len(graph["action"]) != 8:
        raise ValueError("Action encoding")
    for x, mask in zip(entities, observed, strict=True):
        if len(x) != DIMS[domain] or len(mask) != DIMS[domain]:
            raise ValueError("Domain feature dimensions")
        if any(v not in (0, 1) for v in mask) or any(v != 0 for v, m in zip(x, mask) if not m):
            raise ValueError("Unobserved values must be masked and zero")
    if any(len(row) != n or any(len(edge) != 3 for edge in row) for row in relations):
        raise ValueError("Relation dimensions")
    values = (
        [v for row in entities for v in row]
        + [v for row in relations for edge in row for v in edge]
        + graph["action"]
    )
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError("Finite numeric graph required")
    return graph
