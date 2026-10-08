"""Environment-neutral benchmark records and isolated model-facing views."""
from dataclasses import asdict, dataclass
from typing import Any
import copy


@dataclass(frozen=True)
class BenchmarkExample:
    example_id: str
    environment: str
    problem: str
    observation: dict[str, Any]
    initial_state: dict[str, Any]
    goal: dict[str, Any]
    target_answer: Any
    target_trajectory: list[dict[str, Any]]
    metadata: dict[str, Any]

    def view(self, form: str, supervision: bool = False) -> dict:
        result = {"example_id": self.example_id, "environment": self.environment,
                  "problem": self.problem, "observation": self.observation}
        if form == "answer_only":
            if supervision:
                result["answer"] = self.target_answer
        elif form == "trajectory_supervised":
            result.update(initial_state=self.initial_state, goal=self.goal)
            if supervision:
                result.update(answer=self.target_answer, trajectory=self.target_trajectory)
        elif form == "experiential":
            result.update(initial_state=self.initial_state, goal=self.goal)
        else:
            raise ValueError(f"Unknown form: {form}")
        return copy.deepcopy(result)

    def to_dict(self) -> dict:
        return asdict(self)
