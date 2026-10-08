"""Domain-neutral, tagged bounded rollout interfaces."""
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class HypotheticalState:
    value: Any
    hypothetical: bool = True


class TransitionModel(Protocol):
    def predict(self, state: HypotheticalState, action) -> HypotheticalState: ...


class DreamEngine:
    def __init__(self, transition_model, candidates, score, depth=2, branches=3):
        if depth < 1 or branches < 1:
            raise ValueError("Positive rollout bounds required")
        self.model, self.candidates, self.score = transition_model, candidates, score
        self.depth, self.branches = depth, branches

    def rollouts(self, real_state):
        frontier = [(HypotheticalState(real_state), (), 0.0)]
        for _ in range(self.depth):
            following = []
            for state, path, total in frontier:
                actions = list(self.candidates(state.value))[:self.branches]
                if not actions:
                    following.append((state, path, total))
                for action in actions:
                    predicted = self.model.predict(state, action)
                    if not predicted.hypothetical:
                        raise ValueError("Simulation returned a real state")
                    following.append((predicted, path + (action,), total + self.score(state.value, action, predicted.value)))
            frontier = following
        return frontier

    def select(self, real_state):
        paths = self.rollouts(real_state)
        paths = [item for item in paths if item[1]]
        return max(paths, key=lambda item: item[2])[1][0] if paths else None


class FunctionalTransitionModel:
    """Deterministic or learned callable adapter; real environments never mutated."""
    def __init__(self, function): self.function = function
    def predict(self, state, action):
        if not state.hypothetical:
            raise ValueError("Expected hypothetical input")
        return HypotheticalState(self.function(state.value, action))


class LearnedTransitionAdapter:
    def __init__(self, backend): self.backend = backend
    def predict(self, state, action):
        if not state.hypothetical:
            raise ValueError("Expected hypothetical input")
        return HypotheticalState(self.backend.predict_state(state.value, action))
