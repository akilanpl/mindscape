from dataclasses import dataclass

from mindscape.core.schema import EvidenceKind, Trajectory
from mindscape.environments.multiplication.environment import MultiplicationEnvironment


@dataclass(frozen=True)
class Verification:
    trajectory_valid: bool
    final_answer_valid: bool
    goal_reached: bool
    errors: tuple[str, ...] = ()

    @property
    def verified(self) -> bool:
        return self.trajectory_valid and self.final_answer_valid and self.goal_reached


def verify(trajectory: Trajectory, claimed_answer: int | None = None) -> Verification:
    """Replay every field; an incomplete prefix cannot support a final answer."""
    env = MultiplicationEnvironment()
    try:
        initial = env.reset(trajectory.observation)
        if initial != trajectory.initial_state:
            return Verification(False, False, False, ("Initial state mismatch",))
        for index, transition in enumerate(trajectory.transitions):
            if transition.result.kind != EvidenceKind.ACTUAL_RESULT:
                return Verification(False, False, False, (f"Nonactual result at {index}",))
            expected = env.step(transition.action)
            if expected != transition:
                return Verification(False, False, False, (f"Transition mismatch at {index}",))
    except (ValueError, TypeError, AttributeError, IndexError) as exc:
        return Verification(False, False, False, (str(exc),))
    final = env.get_state()
    reached = env.is_goal_reached()
    answer_valid = reached and (claimed_answer is None or
                                (type(claimed_answer) is int and claimed_answer == final.answer))
    errors = () if reached else ("Incomplete trajectory",)
    if reached and not answer_valid:
        errors = ("Claimed answer mismatch",)
    return Verification(True, answer_valid, reached, errors)
