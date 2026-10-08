"""Model-facing generic runner integration with truly external private assessment."""

from mindscape.coding.schema import CodeTask
from mindscape.models.base import Prediction


class CodingBenchmarkModel:
    def __init__(self, policy):
        self.policy = policy
        self.identifier = "coding:" + getattr(policy, "condition", "typed_tools")
        self.parameter_count = policy.coder.parameter_count

    def predict(self, view):
        observation = view["observation"]
        task = CodeTask(
            view["example_id"],
            dict(observation["repository"]),
            view["problem"],
            observation["visible_tests"],
            None,
            {},
            {},
            {},
        )
        row = self.policy.solve(task)
        # Private cases are not present in this object or policy. The coding evaluator
        # assesses the returned repository independently after prediction is complete.
        return Prediction(
            answer=row["repository"],
            trajectory=row["trajectory"],
            model_calls=row["model_calls"],
            diagnostics={"assessment": "deferred", "policy_wall_seconds": row["wall_seconds"]},
        )
