"""Infrastructure reference only; no training or learned benchmark claims."""
from mindscape.data.backends import get_backend
from mindscape.models.base import Prediction


class ReferenceModel:
    identifier = "deterministic_procedural_reference"
    parameter_count = 0

    def predict(self, example):
        answer, trajectory = get_backend(example["environment"]).reference(example)
        return Prediction(answer, trajectory, model_calls=0)
