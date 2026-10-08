from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class Prediction:
    answer: Any = None
    trajectory: dict | None = None
    error_type: str | None = None
    model_calls: int | None = None


class Model(Protocol):
    identifier: str
    parameter_count: int | None
    def predict(self, example: dict) -> Prediction: ...
