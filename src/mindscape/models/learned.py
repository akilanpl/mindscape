"""Learned answer decoder and state-conditioned environment policy wrappers."""
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np

from mindscape.core.schema import Action, Observation
from mindscape.core.serialization import decode
from mindscape.environments.multiplication.environment import MultiplicationEnvironment
from mindscape.memory.working import WorkingMemory
from mindscape.models.base import Prediction
from mindscape.models.encoding import ACTION_NAMES, FORMAT_VERSION, decode_answer, features, serialize_state
from mindscape.models.numpy_backend import NumpyMLP
from mindscape.verification.verifier import verify


class LearnedModel:
    def __init__(self, backend, metadata=None):
        self.backend = backend
        self.training_metadata = metadata or {}

    @property
    def parameter_count(self):
        return self.backend.parameter_count

    def save(self, path):
        self.backend.save(path)
        (Path(path) / "model.json").write_text(json.dumps({"kind": self.kind,
            "format_version": FORMAT_VERSION, "training_metadata": self.training_metadata}, indent=2))

    @staticmethod
    def load(path):
        record = json.loads((Path(path) / "model.json").read_text())
        if record["format_version"] != FORMAT_VERSION:
            raise ValueError("Unsupported feature/serialization version")
        classes = {"baseline": BaselineModel, "mindscape": MindscapeModel}
        return classes[record["kind"]](NumpyMLP.load(path), record["training_metadata"])


class BaselineModel(LearnedModel):
    kind = "baseline"
    identifier = "numpy_mlp_answer_only_v1"

    def predict(self, example):
        labels = self.backend.generate(features(example["observation"]["operands"])[None, :])[0]
        return Prediction(answer=decode_answer(labels), model_calls=1)


class MindscapeModel(LearnedModel):
    kind = "mindscape"
    identifier = "numpy_mlp_trajectory_policy_v1"

    def __init__(self, backend, metadata=None, mask=True, max_steps=256, timeout_seconds=10):
        super().__init__(backend, metadata)
        self.mask = mask
        if not mask:
            self.identifier += "_unmasked"
        self.max_steps = max_steps
        self.timeout_seconds = timeout_seconds
        self.memory = WorkingMemory()
        self.last_episode = None

    def select_action(self, representation, operands):
        scores = self.backend.logits(features(operands, representation["state"], representation["previous_action"])[None, :])[0]
        raw = int(np.argmax(scores))
        if self.mask:
            allowed = {a["name"] for a in representation["valid_actions"]}
            scores = np.where([name in allowed for name in ACTION_NAMES], scores, -np.inf)
            if not np.isfinite(scores).any():
                raise ValueError("No legal action")
        selected = int(np.argmax(scores))
        state = representation["state"]
        return Action(ACTION_NAMES[selected], state["position"], state["row_position"]), ACTION_NAMES[raw]

    def predict(self, example):
        env = MultiplicationEnvironment()
        observation = decode(Observation, example["observation"])
        state = env.reset(observation)
        self.memory.reset(observation, state, env.get_goal())
        started, decisions, error = time.perf_counter(), [], None
        for _ in range(self.max_steps):
            if env.is_goal_reached():
                break
            if time.perf_counter() - started > self.timeout_seconds:
                error = "timeout"
                break
            representation = serialize_state(example["problem"], env.get_state(), env.get_goal(), env.valid_actions(),
                self.memory.action_history[-1].name if self.memory.action_history else None)
            action, raw = self.select_action(representation, observation.operands)
            valid = action in env.valid_actions()
            decision = {"action": asdict(action), "unmasked_action": raw,
                        "action_valid": valid, "masked_correction": action.name != raw,
                        "transition_valid": None}
            decisions.append(decision)
            if not valid:
                error = "invalid_action"
                break
            transition = env.step(action)
            self.memory.update(transition)
            decision["transition_valid"] = verify(env.get_trajectory()).trajectory_valid
            if not decision["transition_valid"]:
                error = "invalid_transition"
                break
        else:
            if not env.is_goal_reached():
                error = "timeout"
        trajectory = env.get_trajectory()
        answer = env.get_state().answer
        outcome = verify(trajectory, answer)
        if error is None and not outcome.verified:
            error = "goal_failure"
        diagnostics = {"decisions": decisions, "mask_enabled": self.mask,
                       "mask_forced_singleton": True, "verification": asdict(outcome)}
        self.last_episode = {"initial_state": asdict(trajectory.initial_state),
                             "trajectory": asdict(trajectory), "final_state": asdict(env.get_state()),
                             "answer": answer, "diagnostics": diagnostics}
        return Prediction(answer, asdict(trajectory), error, len(decisions), diagnostics)


class ExperientialLearnerInterface:
    """Regime C extension point; intentionally no implemented feedback learning."""
    def learn_from_feedback(self, state, action, actual_transition, reward):
        raise NotImplementedError("Regime C is not implemented")
