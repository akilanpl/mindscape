"""Compatible V3 records for continuous physics; preserved integer kernel untouched."""

import json
from dataclasses import asdict

import numpy as np

from mindscape.v3.schema import Concept, Entity, Goal, Observation, Relationship

from .schema import ATTRIBUTES, digest


def represent(public, goal="Predict physical transition"):
    if set(public) != {"values", "mask", "context"}:
        raise ValueError("Unexpected/private observation fields")
    values = np.asarray(public["values"])
    mask = np.asarray(public["mask"])
    entities = []
    missing = []
    relationships = []
    for i, row in enumerate(values):
        if not row[9]:
            continue
        attributes = tuple(
            (name, json.dumps(float(row[j]))) for j, name in enumerate(ATTRIBUTES) if mask[i, j]
        )
        entities.append(Entity(f"object{i}", attributes))
        missing.extend(f"object{i}.{ATTRIBUTES[j]}" for j in range(len(row)) if not mask[i, j])
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            relationships.append(
                Relationship(entities[i].id, "observed_relative_motion", entities[j].id)
            )
    observation = Observation(
        "physics_world",
        tuple(
            float(v) for v in np.concatenate((values.flatten(), mask.flatten(), public["context"]))
        ),
        Goal(goal),
        tuple(entities),
        tuple(relationships),
        bool(missing),
        tuple(missing),
    )
    return observation


def concept_catalog():
    descriptions = (
        "observable acceleration",
        "relative motion",
        "possible contact",
        "momentum proxy when mass is visible",
        "material-dependent response",
    )
    return [
        asdict(Concept(digest(description), "physics_world", description))
        for description in descriptions
    ]
