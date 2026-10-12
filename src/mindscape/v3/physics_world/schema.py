"""Public observations/actions and private scene records; canonical identities."""

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np

MAX_OBJECTS = 3
ATTRIBUTES = (
    "x",
    "y",
    "vx",
    "vy",
    "mass",
    "radius",
    "friction",
    "restitution",
    "contact",
    "active",
)
STATE_DIM = len(ATTRIBUTES)
HISTORY = 4
DT = 1 / 30


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class ObjectSpec:
    position: tuple[float, float]
    velocity: tuple[float, float]
    mass: float
    radius: float
    friction: float
    restitution: float


@dataclass(frozen=True)
class Scene:
    family: str
    split: str
    index: int
    seed: int
    mechanism: str
    objects: tuple[ObjectSpec, ...]
    gravity: float
    noise: float
    hidden_parameters: bool
    velocity_dropout: bool
    spring: float
    horizon: int
    ood_condition: str = "none"

    @property
    def identity(self):
        return digest(asdict(self))

    def record(self):
        return asdict(self)

    @classmethod
    def from_record(cls, row):
        row = dict(row)
        row["objects"] = tuple(
            ObjectSpec(
                tuple(o["position"]),
                tuple(o["velocity"]),
                o["mass"],
                o["radius"],
                o["friction"],
                o["restitution"],
            )
            for o in row["objects"]
        )
        return cls(**row)


@dataclass(frozen=True)
class Action:
    forces: tuple[tuple[float, float], ...]
    impulses: tuple[tuple[float, float], ...]

    def vector(self):
        values = np.zeros((MAX_OBJECTS, 4), dtype=np.float32)
        for i, (force, impulse) in enumerate(zip(self.forces, self.impulses, strict=True)):
            values[i] = [*force, *impulse]
        return values

    @classmethod
    def from_vector(cls, value):
        x = np.asarray(value, dtype=np.float64)
        if x.shape != (MAX_OBJECTS, 4) or not np.isfinite(x).all() or abs(x).max() > 20:
            raise ValueError("Action bounds")
        return cls(tuple(tuple(v[:2]) for v in x), tuple(tuple(v[2:]) for v in x))


def observe(state, scene, step):
    """No future or private parameters returned. Fresh deterministic sensor noise."""
    value = np.asarray(state, dtype=np.float32).copy()
    mask = np.ones_like(value)
    if scene.hidden_parameters:
        value[:, [4, 6, 7]] = 0
        mask[:, [4, 6, 7]] = 0
    if scene.velocity_dropout:
        value[:, 2:4] = 0
        mask[:, 2:4] = 0
    rng = np.random.default_rng(int(digest([scene.identity, "sensor", step])[:16], 16))
    value[:, :4] += rng.normal(0, scene.noise, value[:, :4].shape).astype(np.float32) * mask[:, :4]
    value[value[:, 9] == 0] = 0
    mask[value[:, 9] == 0] = 0
    # Gravity and spring are permitted context; hidden parameters are mass/materials.
    context = np.asarray([scene.gravity, DT, scene.spring, step * DT], dtype=np.float32)
    return {"values": value, "mask": mask, "context": context}
