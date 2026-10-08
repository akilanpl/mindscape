"""Numerical proposal learners; primary inference has no oracle correction channel."""
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np

from mindscape.core.schema import Observation
from mindscape.core.serialization import decode
from mindscape.environments.multiplication.claims import ClaimAction, ClaimsEnvironment, apply_claim
from mindscape.memory.working import WorkingMemory
from mindscape.models.base import Prediction
from mindscape.models.encoding import decode_answer
from mindscape.models.numpy_backend import NumpyMLP
from mindscape.models.study_encoding import VERSION, answer_features, candidate_scores, policy_features
from mindscape.reasoning.simulator import DreamEngine, FunctionalTransitionModel


class StudyModel:
    def __init__(self, backend, condition, metadata=None, constrained=False, dream=True,
                 no_state=False, no_relation=False, no_goal=False, no_interaction=False,
                 perturb=False):
        self.backend, self.condition = backend, condition
        self.training_metadata = metadata or {}
        self.identifier = "claims_" + condition
        self.constrained, self.dream = constrained, dream
        self.no_state, self.no_relation, self.no_goal = no_state, no_relation, no_goal
        self.no_interaction, self.perturb = no_interaction, perturb
        self.memory = WorkingMemory()

    @property
    def parameter_count(self): return self.backend.parameter_count

    def save(self, path):
        self.backend.save(path)
        (Path(path) / "study_model.json").write_text(json.dumps({"condition": self.condition,
            "format_version": VERSION, "metadata": self.training_metadata,
            "options": {key: getattr(self, key) for key in ("constrained", "dream", "no_state", "no_relation", "no_goal", "no_interaction", "perturb")}}, indent=2))

    @classmethod
    def load(cls, path):
        record = json.loads((Path(path) / "study_model.json").read_text())
        if record["format_version"] != VERSION:
            raise ValueError("Unknown study serialization version")
        return cls(NumpyMLP.load(path), record["condition"], record["metadata"], **record["options"])

    def predict(self, view):
        if self.condition in ("answer_only", "structured"):
            if self.perturb:
                view = json.loads(json.dumps(view, sort_keys=True))
                view["irrelevant_note"] = "blue square"
            labels = self.backend.generate(answer_features(view, self.condition == "structured")[None, :])[0]
            return Prediction(decode_answer(labels), model_calls=1)
        observation = decode(Observation, view["observation"])
        env = ClaimsEnvironment()
        state = env.reset(observation)
        self.memory.reset(observation, state, env.get_goal())
        decisions, calls, error = [], 0, None
        while env.state.phase != "done":
            if len(decisions) >= 256:
                error = "timeout"
                break
            cache = {}
            def scores(s):
                nonlocal calls
                # Canonical keys and ignored distractor demonstrate serialization-order invariance.
                representation = {"version": VERSION, "state": asdict(s), "goal": asdict(s.goal)}
                if self.perturb:
                    representation = dict(reversed(list(representation.items())))
                    representation["distractor"] = "blue square"
                key = json.dumps(representation["state"], sort_keys=True)
                if key not in cache:
                    cache[key] = candidate_scores(self.backend,
                        policy_features(s, self.no_state, self.no_relation, self.no_goal))
                    calls += 1
                return cache[key]
            raw_scores = scores(env.state)
            count = 100 if self.constrained else 101
            value = int(np.argmax(raw_scores[:count]))
            simulated = []
            if self.dream and value < 100:
                def candidates(s):
                    if s.phase == "done": return []
                    return [ClaimAction(int(v), s.position, s.row_position)
                            for v in np.argsort(-scores(s)[:100])[:3]]
                def score(s, action, following):
                    z = scores(s)[:100]
                    shifted = z - z.max()
                    return float(shifted[action.value] - np.log(np.exp(shifted).sum()))
                planner = DreamEngine(FunctionalTransitionModel(apply_claim), candidates, score, depth=2, branches=3)
                rollouts = planner.rollouts(env.state)
                best = max(rollouts, key=lambda item: item[2])
                value = best[1][0].value
                simulated = [{"hypothetical": True, "state": asdict(best[0].value),
                              "candidate_actions": [asdict(a) for a in best[1]], "score": best[2]}]
            decision = {"candidate_action_count": count, "selected_value": value,
                        "model_prediction": value, "action_valid": value < 100,
                        "hypothetical_rollouts": simulated}
            decisions.append(decision)
            if value == 100:
                error = "invalid_action"
                break
            action = ClaimAction(value, env.state.position, env.state.row_position)
            if self.no_interaction:
                env.state = apply_claim(env.state, action)
                self.memory.current_state = env.state
                decision["hypothetical_result"] = {"value": value, "hypothetical": True}
            else:
                transition = env.step(action)  # Executes claim even when wrong; never corrects it.
                self.memory.update(transition)
                decision["actual_result"] = asdict(transition.result)
            # No correctness signal is passed back to scores or model-visible memory.
        trajectory = asdict(env.trajectory)
        answer = env.state.answer
        if self.no_interaction:
            trajectory = None  # Open-loop model recurrence, no supporting environment evidence.
        return Prediction(answer, trajectory, error, calls,
            {"decisions": decisions, "oracle_feedback_visible": False,
             "execution_mode": "constrained" if self.constrained else "unconstrained",
             "dream_model": "unverified_proposed_state_dynamics" if self.dream else None})
