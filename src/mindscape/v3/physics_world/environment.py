"""Pinned authoritative CPU simulator. Never imported by neural model/planner."""

import numpy as np
import pymunk

from .schema import DT, MAX_OBJECTS, STATE_DIM, Action

ENGINE_VERSION = "7.2.0"
ENGINE_REVISION = "2.0.1-ade7ed72849e60289eefb7a41e79ae6322fefaf3"


class Environment:
    def __init__(self, scene):
        if pymunk.version != ENGINE_VERSION or pymunk.chipmunk_version != ENGINE_REVISION:
            raise RuntimeError("Unpinned physics engine")
        self.scene = scene
        self.steps = 0
        self.space = pymunk.Space(threaded=False)
        self.space.gravity = (0, -scene.gravity)
        self.space.iterations = 20
        self.bodies = []
        self.shapes = []
        self.contacts = set()
        self.space.on_collision(None, None, post_solve=self._contact)
        floor = pymunk.Segment(self.space.static_body, (-100, 0), (100, 0), 0.02)
        floor.friction = 1
        floor.elasticity = 1
        self.space.add(floor)
        for i, spec in enumerate(scene.objects):
            body = pymunk.Body(spec.mass, pymunk.moment_for_circle(spec.mass, 0, spec.radius))
            body.position = spec.position
            body.velocity = spec.velocity
            shape = pymunk.Circle(body, spec.radius)
            shape.friction = spec.friction
            shape.elasticity = spec.restitution
            shape.collision_type = i + 1
            self.space.add(body, shape)
            self.bodies.append(body)
            self.shapes.append(shape)
        if scene.spring:
            spring = pymunk.DampedSpring(
                self.bodies[0], self.bodies[1], (0, 0), (0, 0), 1.5, scene.spring, 0.3
            )
            spring.collide_bodies = True
            self.space.add(spring)

    def _contact(self, arbiter, space, data):
        for shape in arbiter.shapes:
            if shape.collision_type:
                self.contacts.add(shape.collision_type - 1)

    def state(self):
        x = np.zeros((MAX_OBJECTS, STATE_DIM), dtype=np.float64)
        for i, (body, spec) in enumerate(zip(self.bodies, self.scene.objects, strict=True)):
            x[i] = [
                *body.position,
                *body.velocity,
                spec.mass,
                spec.radius,
                spec.friction,
                spec.restitution,
                float(i in self.contacts),
                1,
            ]
        if not np.isfinite(x).all():
            raise RuntimeError("Nonfinite physics state")
        return x

    def step(self, action):
        if not isinstance(action, Action):
            raise TypeError("Typed action required")
        vector = action.vector()
        if not np.isfinite(vector).all() or abs(vector).max() > 20:
            raise ValueError("Action bounds")
        self.contacts.clear()
        for i, body in enumerate(self.bodies):
            body.apply_impulse_at_world_point(tuple(vector[i, 2:]), body.position)
        for _ in range(4):
            for i, body in enumerate(self.bodies):
                body.apply_force_at_world_point(tuple(vector[i, :2]), body.position)
            self.space.step(DT / 4)
        self.steps += 1
        return self.state()

    def energy(self):
        state = self.state()
        kinetic = sum(0.5 * r[4] * (r[2] ** 2 + r[3] ** 2) for r in state if r[9])
        potential = sum(r[4] * self.scene.gravity * r[1] for r in state if r[9])
        if self.scene.spring:
            extension = np.linalg.norm(state[0, :2] - state[1, :2]) - 1.5
            potential += 0.5 * self.scene.spring * extension**2
        return float(kinetic + potential)


def trajectory(scene):
    from .schema import HISTORY, observe

    rng = np.random.default_rng(int(scene.identity[:16], 16))
    env = Environment(scene)
    states = [env.state()]
    actions = []
    observations = [observe(states[0], scene, 0)]
    for step in range(HISTORY + scene.horizon):
        vector = np.zeros((MAX_OBJECTS, 4), dtype=np.float32)
        vector[: len(scene.objects), :2] = rng.uniform(-0.4, 0.4, (len(scene.objects), 2))
        if step % 12 == 0:
            vector[: len(scene.objects), 2:] = rng.uniform(-0.3, 0.3, (len(scene.objects), 2))
        state = env.step(Action.from_vector(vector))
        states.append(state)
        actions.append(vector)
        observations.append(observe(state, scene, step + 1))
    return {
        "identity": scene.identity,
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(actions),
        "observations": np.asarray([o["values"] for o in observations]),
        "masks": np.asarray([o["mask"] for o in observations]),
        "contexts": np.asarray([o["context"] for o in observations]),
        "initial_energy": float(
            sum(
                0.5 * o.mass * sum(v * v for v in o.velocity)
                + o.mass * scene.gravity * o.position[1]
                for o in scene.objects
            )
        ),
    }
