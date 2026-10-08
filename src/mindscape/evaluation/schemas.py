from dataclasses import dataclass
from enum import Enum
from typing import Any


class ErrorType(str, Enum):
    CORRECT = "correct"
    ARITHMETIC_ERROR = "arithmetic_error"
    INVALID_ACTION = "invalid_action"
    INVALID_TRANSITION = "invalid_transition"
    MALFORMED_TRAJECTORY = "malformed_trajectory"
    GOAL_FAILURE = "goal_failure"
    UNSUPPORTED_ANSWER = "unsupported_answer"
    TIMEOUT = "timeout"
    MODEL_ERROR = "model_error"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class PredictionRecord:
    example_id: str
    predicted_answer: Any
    target_answer: Any
    correct: bool
    predicted_trajectory: dict | None
    trajectory_valid: bool | None
    goal_reached: bool
    grounded: bool
    error_type: str


@dataclass(frozen=True)
class ExperimentResult:
    experiment_id: str
    timestamp: str
    environment: str
    model: str
    regime: str
    dataset_size: int
    seed: int
    split: str
    accuracy: float | None
    ood_accuracy: float | None
    grounded_rate: float | None
    trajectory_validity: float | None
    goal_success_rate: float | None
    unsupported_rate: float | None
    training_time: float | None
    inference_time: float
    model_calls: int | None
    parameter_count: int | None
    notes: str
