"""Authoritative decimal digit/carry multiplication transitions."""
from dataclasses import replace

from mindscape.core.schema import (
    Action, Entity, Event, Goal, Observation, Relation, Result, State, Trajectory, Transition,
)


def digits(value: int) -> tuple[int, ...]:
    return tuple(int(d) for d in reversed(str(abs(value))))


class MultiplicationEnvironment:
    def __init__(self):
        self._state: State | None = None
        self._trajectory: Trajectory | None = None

    def reset(self, problem: tuple[int, int] | Observation) -> State:
        observation = problem if isinstance(problem, Observation) else Observation(problem)
        a, b = observation.operands
        state = State(
            entities=(Entity("left", a), Entity("right", b)),
            relations=(Relation("left", "multiplied_by", "right"),),
            left_digits=digits(a), right_digits=digits(b),
            sign=-1 if (a < 0) != (b < 0) else 1,
        )
        self._state = state
        self._trajectory = Trajectory(observation, state)
        return state

    def get_state(self) -> State:
        if self._state is None:
            raise RuntimeError("Call reset before using the environment")
        return self._state

    def get_goal(self) -> Goal:
        return self.get_state().goal

    def valid_actions(self) -> tuple[Action, ...]:
        s = self.get_state()
        if s.phase == "done":
            return ()
        return (Action(s.phase, s.position, s.row_position),)

    def step(self, action: Action) -> Transition:
        s = self.get_state()
        if action not in self.valid_actions():
            raise ValueError("Illegal or stale action")
        if action.name == "multiply":
            raw = s.left_digits[s.position] * s.right_digits[s.row_position] + s.carry
            digit, carry = raw % 10, raw // 10
            position = s.position + 1
            after = replace(s, position=position, carry=carry,
                            row_value=s.row_value + digit * 10 ** s.position,
                            phase="flush" if position == len(s.left_digits) else "multiply")
            event = Event("digit_written", "Multiply operand digits, add incoming carry")
            result = Result(raw, digit, carry)
        elif action.name == "flush":
            row = s.row_value + s.carry * 10 ** s.position
            after = replace(s, row_value=row, carry=0, phase="accumulate")
            event = Event("carry_flushed", "Append remaining carry to the partial row")
            result = Result(row)
        elif action.name == "accumulate":
            total = s.accumulated + s.row_value * 10 ** s.row_position
            row_position = s.row_position + 1
            after = replace(s, accumulated=total, row_position=row_position,
                            position=0, row_value=0,
                            phase="finish" if row_position == len(s.right_digits)
                            else "multiply")
            event = Event("row_accumulated", "Add place-shifted partial row")
            result = Result(total)
        elif action.name == "finish":
            answer = s.sign * s.accumulated
            after = replace(s, answer=answer, phase="done")
            event = Event("answer_materialized", "Apply the operand sign")
            result = Result(answer)
        else:
            raise ValueError("Unknown action")
        transition = Transition(s, action, event, result, after)
        trajectory = self.get_trajectory()
        self._trajectory = replace(trajectory, transitions=trajectory.transitions + (transition,))
        self._state = after
        return transition

    def is_goal_reached(self) -> bool:
        s = self.get_state()
        a, b = self.get_trajectory().observation.operands
        return s.phase == "done" and s.answer == a * b

    def get_trajectory(self) -> Trajectory:
        if self._trajectory is None:
            raise RuntimeError("Call reset before using the environment")
        return self._trajectory

    def run(self) -> Trajectory:
        while self.valid_actions():
            self.step(self.valid_actions()[0])
        return self.get_trajectory()
