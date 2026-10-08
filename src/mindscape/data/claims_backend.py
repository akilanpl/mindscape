from dataclasses import asdict
from mindscape.core.schema import Observation, State
from mindscape.core.serialization import decode
from mindscape.data.backends import MultiplicationBackend, stable_hash
from mindscape.data.schemas import BenchmarkExample
from mindscape.environments.multiplication.claims import ClaimTrajectory, ClaimsEnvironment, expected_value, verify_claims


class ClaimsBackend(MultiplicationBackend):
    name = "integer_multiplication_claims"
    version = "claims-1.0.0"

    def example(self, observation, seed, split):
        env = ClaimsEnvironment()
        env.reset(decode(Observation, observation))
        trajectory = env.reference()
        category = self.category(observation)
        ts = trajectory.transitions
        return BenchmarkExample(stable_hash([self.name, self.identity(observation)]), self.name,
            " × ".join(map(str, observation["operands"])), observation,
            asdict(trajectory.initial_state), asdict(env.get_goal()), env.state.answer,
            [asdict(t) for t in ts],
            {"random_seed": seed, "operand_sizes": [int(x) for x in category.split("x")],
             "structural_category": category, "split": split, "generator_version": self.version,
             "difficulty": {"carry_count": sum(t.result.carry > 0 for t in ts),
                 "step_count": len(ts), "multiplication_steps": sum(t.state_before.phase == "compute_claim" for t in ts),
                 "max_operand_digits": max(len(str(abs(x))) for x in observation["operands"]),
                 "max_intermediate_value": max(t.result.value for t in ts)}})

    def assess(self, example, answer, trajectory):
        if trajectory is None:
            return None, False, "unsupported_answer"
        try:
            typed = decode(ClaimTrajectory, trajectory)
            if stable_hash(asdict(typed.observation)) != stable_hash(example.observation):
                return False, False, "invalid_transition"
            valid, goal = verify_claims(typed, answer)
            return valid, goal, "correct" if goal else "invalid_transition" if not valid else "goal_failure"
        except (ValueError, TypeError, KeyError, AttributeError):
            return False, False, "malformed_trajectory"

    def reference(self, view):
        env = ClaimsEnvironment()
        env.reset(decode(Observation, view["observation"]))
        trajectory = env.reference()
        return env.state.answer, asdict(trajectory)


    def annotate(self, trajectory, diagnostics):
        if not diagnostics:
            return
        if trajectory is None:
            # Posthoc hypothetical judgments do not create real execution evidence.
            for decision in diagnostics.get("decisions", []):
                if "state_before" in decision:
                    state = decode(State, decision["state_before"])
                    decision["correct_action"] = expected_value(state)
                    decision["hypothetical_judgment"] = decision["selected_value"] == decision["correct_action"]
                    decision["transition_valid"] = None
            return
        try:
            typed = decode(ClaimTrajectory, trajectory)
            for decision, transition in zip(diagnostics.get("decisions", []), typed.transitions):
                decision["correct_action"] = expected_value(transition.state_before)
                decision["verifier_judgment"] = transition.action.value == decision["correct_action"]
                decision["transition_valid"] = decision["verifier_judgment"]
        except (ValueError, TypeError, KeyError):
            diagnostics["verification_error"] = "malformed_trajectory"


from mindscape.data.backends import register
register(ClaimsBackend())
