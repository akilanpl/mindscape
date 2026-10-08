import json
from pathlib import Path
import random

from mindscape.data.backends import get_backend, stable_hash
from mindscape.data.schemas import BenchmarkExample
from mindscape.data.splits import nested_subsets, validate_dataset


def generate(config):
    backend = get_backend(config["environment"])
    specs = config["splits"]
    if not {"train", "validation", "test"}.issubset(specs):
        raise ValueError("train, validation and test are required")
    train = {backend.structure_identity(c) for c in backend.structures(specs["train"])}
    ood = {backend.structure_identity(c) for c in (backend.structures(specs["ood_test"]) if "ood_test" in specs else [])}
    if train & ood:
        raise ValueError("Overlapping OOD structures")
    seen, result = set(), {}
    for split in ["train", "validation", "test"] + sorted(set(specs) - {"train", "validation", "test"}):
        spec = specs[split]
        categories, count = backend.structures(spec), spec["count"]
        if type(count) is not int or count < 0 or not categories or len(set(categories)) != len(categories):
            raise ValueError("Invalid count or structures")
        if split == "train" and count < len(categories):
            raise ValueError("Training needs at least one example per structure")
        rng = random.Random(f"{config['seed']}:{split}")
        examples = []
        attempts = 0
        while len(examples) < count:
            attempts += 1
            if attempts > max(10000, count * 200):
                raise ValueError("Unique sample space exhausted; reduce counts or broaden structures")
            category = categories[len(examples)] if split == "train" and len(examples) < len(categories) else rng.choice(categories)
            observation = backend.sample(category, rng, config)
            identity = backend.identity(observation)
            if identity in seen:
                continue
            example = backend.example(observation, config["seed"], split)
            accepted = True
            for metric, bounds in config.get("difficulty_bounds", {}).items():
                value = example.metadata["difficulty"][metric]
                if bounds.get("min", float("-inf")) > bounds.get("max", float("inf")):
                    raise ValueError("Invalid difficulty bounds")
                if not bounds.get("min", value) <= value <= bounds.get("max", value):
                    accepted = False
            if not accepted:
                continue
            seen.add(identity)
            examples.append(example)
        result[split] = examples
    validate_dataset(result, config)
    nested_subsets(result["train"], config.get("budgets", []), config["seed"])
    return result


def save_dataset(path, splits, config):
    validate_dataset(splits, config)
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    manifest = {"dataset_version": "1.0.0", "config": config, "files": {}}
    for split, examples in splits.items():
        content = "".join(json.dumps(e.to_dict(), sort_keys=True, separators=(",", ":")) + "\n" for e in examples)
        (path / f"{split}.jsonl").write_text(content)
        manifest["files"][split] = stable_hash([e.to_dict() for e in examples])
    subsets = nested_subsets(splits["train"], config.get("budgets", []), config["seed"])
    manifest["nested_subsets"] = {str(n): [e.example_id for e in subset] for n, subset in subsets.items()}
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return path


def load_dataset(path):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["dataset_version"] != "1.0.0":
        raise ValueError("Unsupported dataset version")
    splits = {}
    for split, digest in manifest["files"].items():
        rows = [json.loads(line) for line in (path / f"{split}.jsonl").read_text().splitlines()]
        if stable_hash(rows) != digest:
            raise ValueError("Dataset checksum mismatch")
        splits[split] = [BenchmarkExample(**row) for row in rows]
    config = manifest["config"]
    validate_dataset(splits, config)
    expected = {str(n): [e.example_id for e in subset] for n, subset in
                nested_subsets(splits["train"], config.get("budgets", []), config["seed"]).items()}
    if manifest["nested_subsets"] != expected:
        raise ValueError("Corrupt training subsets")
    return splits, manifest
