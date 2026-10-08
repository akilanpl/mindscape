"""Training orchestration independent of environments and inference wrappers."""
import json
from pathlib import Path
import platform
import subprocess
import time
from datetime import datetime, timezone
import numpy as np

from mindscape.data.backends import stable_hash
from mindscape.data.generation import load_dataset
from mindscape.models.encoding import ANSWER_DIGITS, FORMAT_VERSION
from mindscape.models.learned import BaselineModel, LearnedModel, MindscapeModel
from mindscape.models.numpy_backend import NumpyMLP
from mindscape.training.adapters import arrays
from mindscape.training.optimizer import fit


def validate_config(config):
    required = {"kind", "seed", "budget", "hidden", "steps", "batch_size", "learning_rate", "validation_every"}
    if set(config) != required or config["kind"] not in ("baseline", "mindscape"):
        raise ValueError("Invalid training config fields/regime")
    for key in required - {"kind", "learning_rate"}:
        minimum = 0 if key == "seed" else 1
        if type(config[key]) is not int or config[key] < minimum:
            raise ValueError(f"Invalid {key}")
    if not 0 < config["learning_rate"] < 1:
        raise ValueError("Invalid learning rate")


def train(dataset, config, output):
    validate_config(config)
    splits, manifest = load_dataset(dataset)
    if manifest["config"]["environment"] != "integer_multiplication":
        raise ValueError("First learned feature adapter supports multiplication only")
    ids = manifest["nested_subsets"].get(str(config["budget"]))
    if ids is None:
        raise ValueError("Budget unavailable in dataset manifest")
    lookup = {e.example_id: e for e in splits["train"]}
    examples = [lookup[key] for key in ids]
    train_x, train_y = arrays(examples, config["kind"])
    val_x, val_y = arrays(splits["validation"], config["kind"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    groups = [10] * ANSWER_DIGITS + [2] if config["kind"] == "baseline" else [4]
    backend = NumpyMLP(groups, config["hidden"], config["seed"])
    started = time.perf_counter()
    logs, metrics = fit(backend, train_x, train_y, val_x, val_y,
        config["steps"], config["batch_size"], config["learning_rate"], config["seed"], config["validation_every"])
    elapsed = time.perf_counter() - started
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    metadata = {"config": config, "training_time": elapsed, "dataset_hash": stable_hash(manifest),
                "dataset_version": manifest["dataset_version"], "training_ids": ids,
                "training_rows": len(train_x), "validation_rows": len(val_x),
                "regime": "answer_only" if config["kind"] == "baseline" else "trajectory_supervised",
                "checkpoint": str((output / "checkpoint").resolve()), "parameter_count": backend.parameter_count,
                "timestamp": datetime.now(timezone.utc).isoformat(), "format_version": FORMAT_VERSION,
                "software": {"python": platform.python_version(), "numpy": np.__version__, "git_revision": revision},
                "hardware": {"machine": platform.machine(), "processor": platform.processor()},
                "validation_metrics": metrics}
    model = (BaselineModel if config["kind"] == "baseline" else MindscapeModel)(backend, metadata)
    model.save(output / "checkpoint")
    (output / "config.json").write_text(json.dumps({"training": config, "dataset_manifest": manifest}, indent=2))
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (output / "metrics.json").write_text(json.dumps({**metrics, "training_time": elapsed}, indent=2))
    (output / "training.jsonl").write_text("".join(json.dumps(row) + "\n" for row in logs))
    return model, metadata
