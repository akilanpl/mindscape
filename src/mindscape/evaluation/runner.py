"""Generic, immutable-artifact evaluation; model never receives held-out targets."""
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time
import uuid

from mindscape import __version__
from mindscape.data.backends import get_backend, stable_hash
from mindscape.data.generation import load_dataset
from mindscape.evaluation.evaluator import evaluate
from mindscape.evaluation.metrics import aggregate
from mindscape.evaluation.schemas import ExperimentResult
from mindscape.models.base import Prediction


def run(model, dataset, output_root, split="test", regime="experiential",
        training_size=None, save_predictions=True, notes="", evaluator=evaluate):
    splits, manifest = load_dataset(dataset)  # Always validate ALL splits before model calls.
    config = manifest["config"]
    if split not in ("validation", "test", "ood_test") or split not in splits:
        raise ValueError("Choose an available held-out evaluation split")
    if training_size is not None and str(training_size) not in manifest["nested_subsets"]:
        raise ValueError("Training budget not present in manifest")
    examples = splits[split]
    backend = get_backend(config["environment"])
    records, calls = [], []
    started = time.perf_counter()
    for example in examples:
        view = example.view(regime, supervision=False)
        try:
            prediction = model.predict(view)
            if not isinstance(prediction, Prediction):
                raise TypeError("Model must return Prediction")
        except TimeoutError:
            prediction = Prediction(error_type="timeout")
        except Exception:
            prediction = Prediction(error_type="model_error")
        records.append(evaluator(example, prediction, backend))
        calls.append(prediction.model_calls)
    elapsed = time.perf_counter() - started
    metrics = aggregate(records)
    timestamp = datetime.now(timezone.utc).isoformat()
    experiment_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex
    result = ExperimentResult(
        experiment_id, timestamp, backend.name, model.identifier, regime,
        len(splits["train"]) if training_size is None else training_size, config["seed"], split,
        metrics["accuracy"], metrics["accuracy"] if split == "ood_test" else None,
        metrics["grounded_rate"], metrics["trajectory_validity"], metrics["goal_success_rate"],
        metrics["unsupported_rate"], None, elapsed,
        sum(calls) if all(type(n) is int for n in calls) else None,
        getattr(model, "parameter_count", None), notes)
    folder = Path(output_root) / experiment_id
    folder.mkdir(parents=True, exist_ok=False)
    resolved = {"dataset_manifest": manifest, "dataset_hash": stable_hash(manifest),
                "evaluation": {"split": split, "regime": regime, "training_size": training_size,
                               "save_predictions": save_predictions},
                "software": {"mindscape": __version__, "python": platform.python_version()},
                "hardware": {"system": platform.system(), "machine": platform.machine(),
                             "processor": platform.processor()},
                "checkpoint": None, "hyperparameters": None}
    (folder / "config.json").write_text(json.dumps(resolved, indent=2, sort_keys=True))
    payload = {**asdict(result), "example_count": metrics["example_count"],
               "trajectory_count": metrics["trajectory_count"]}
    (folder / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    if save_predictions:
        (folder / "predictions.jsonl").write_text("".join(json.dumps(asdict(r), sort_keys=True) + "\n" for r in records))
    (folder / "summary.md").write_text(
        f"# {experiment_id}\n\nModel: {model.identifier}; split: {split}; examples: {len(examples)}.\n\n"
        + "\n".join(f"- {k}: {v}" for k, v in metrics.items())
        + "\n\nInfrastructure evaluation only; no learned-performance or data-efficiency claim.\n")
    return folder, payload
