from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class Prediction:
    answer: Any = None
    trajectory: dict | None = None
    error_type: str | None = None
    model_calls: int | None = None
    diagnostics: dict | None = None


class Model(Protocol):
    identifier: str
    parameter_count: int | None
    def predict(self, example: dict) -> Prediction: ...


class ModelBackend(Protocol):
    identifier: str
    parameter_count: int
    def logits(self, features): ...
    def generate(self, features): ...
    def save(self, path): ...
    @classmethod
    def load(cls, path): ...
