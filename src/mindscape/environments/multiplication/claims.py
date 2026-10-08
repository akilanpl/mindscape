"""Proposed-value arithmetic: execution never corrects a learner's numerical claim."""
from dataclasses import asdict, dataclass, replace

from mindscape.core.schema import Entity, Event, Goal, Observation, Relation, Result, State
from mindscape.environments.multiplication.environment import digits


@dataclass(frozen=True)
class ClaimAction:
    value: int
    position: int
    row_position: int
    name: str = "write_claim"


@dataclass(frozen=True)
class ClaimTransition:
    state_before: State
    action: ClaimAction
    event: Event
    result: Result
    state_after: State


@dataclass(frozen=True)
class ClaimTrajectory:
    observation: Observation
    initial_state: State
    transitions: tuple[ClaimTransition, ...] = ()


def local_inputs(state):
    if state.phase == "flush_claim":
        return 0, 0, 0, state.carry
    i, j = state.row_position, state.position
    return state.left_digits[j], state.right_digits[i], state.accumulated // 10 ** (i + j) % 10, state.carry


def expected_value(state):
    """Private oracle for target generation and verification, never model features."""
    a, b, existing, carry = local_inputs(state)
    return a * b + existing + carry


def apply_claim(state, action):
    if state.phase == "done" or type(action.value) is not int or not 0 <= action.value < 100:
        raise ValueError("Invalid claim syntax")
    if (action.position, action.row_position) != (state.position, state.row_position):
        raise ValueError("Stale claim")
    if action.name != "write_claim":
        raise ValueError("Unknown claim action")
    offset = state.position + state.row_position
    old = state.accumulated // 10 ** offset % 10
    total = state.accumulated + (action.value % 10 - old) * 10 ** offset
    if state.phase == "compute_claim":
        pos = state.position + 1
        return replace(state, accumulated=total, carry=action.value // 10, position=pos,
                       phase="flush_claim" if pos == len(state.left_digits) else "compute_claim")
    row = state.row_position + 1
    terminal = row == len(state.right_digits)
    return replace(state, accumulated=total, carry=0, position=0, row_position=row,
                   phase="done" if terminal else "compute_claim",
                   answer=state.sign * total if terminal else None)


class ClaimsEnvironment:
    def reset(self, observation):
        if not isinstance(observation, Observation):
            observation = Observation(tuple(observation))
        a, b = observation.operands
        self.observation = observation
        self.state = State((Entity("left", a), Entity("right", b)),
            (Relation("left", "multiplied_by", "right"),), digits(a), digits(b),
            -1 if (a < 0) != (b < 0) else 1, phase="compute_claim")
        self.trajectory = ClaimTrajectory(observation, self.state)
        return self.state

    def get_state(self): return self.state
    def get_goal(self): return self.state.goal
    def get_trajectory(self): return self.trajectory

    def valid_actions(self):
        # Syntactic legality only: never reveal which numerical value is correct.
        return tuple(ClaimAction(value, self.state.position, self.state.row_position)
                     for value in range(100)) if self.state.phase != "done" else ()

    def step(self, action):
        before = self.state
        after = apply_claim(before, action)
        transition = ClaimTransition(before, action,
            Event("claimed_digits_written", "Write the proposed units digit and carry; no correction"),
            Result(action.value, action.value % 10, action.value // 10), after)
        self.state = after
        self.trajectory = replace(self.trajectory, transitions=self.trajectory.transitions + (transition,))
        return transition

    def is_goal_reached(self):
        a, b = self.observation.operands
        return self.state.phase == "done" and self.state.answer == a * b

    def reference(self):
        while self.valid_actions():
            self.step(ClaimAction(expected_value(self.state), self.state.position, self.state.row_position))
        return self.trajectory


def verify_claims(trajectory, answer=None):
    env = ClaimsEnvironment()
    if env.reset(trajectory.observation) != trajectory.initial_state:
        return False, False
    try:
        for transition in trajectory.transitions:
            if transition.action.value != expected_value(env.state):
                return False, False
            if env.step(transition.action) != transition:
                return False, False
    except (ValueError, AttributeError, IndexError):
        return False, False
    reached = env.is_goal_reached()
    return True, reached and type(answer) is int and answer == env.state.answer
