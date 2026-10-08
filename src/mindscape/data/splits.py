"""Identity-aware generation and explicit leakage validation."""
import random
from mindscape.data.backends import get_backend, stable_hash


def validate_dataset(splits, config):
    backend = get_backend(config["environment"])
    if set(splits) != set(config["splits"]):
        raise ValueError("Split set differs from configuration")
    ids, identities, problems, trajectories = set(), set(), set(), set()
    train_categories = set(backend.structures(config["splits"]["train"]))
    heldout = set(backend.structures(config["splits"]["ood_test"])) if "ood_test" in config["splits"] else set()
    if {backend.structure_identity(s) for s in train_categories} & {backend.structure_identity(s) for s in heldout}:
        raise ValueError("OOD structural leakage, including swapped structures")
    observed_train = {backend.category(e.observation) for e in splits["train"]}
    for split, examples in splits.items():
        spec = config["splits"][split]
        if len(examples) != spec["count"]:
            raise ValueError("Incorrect split size")
        for e in examples:
            if e.environment != backend.name or e.metadata["split"] != split:
                raise ValueError("Wrong environment or split membership")
            if e.metadata["random_seed"] != config["seed"]:
                raise ValueError("Example seed differs from dataset seed")
            backend.validate(e)
            category = backend.category(e.observation)
            if category not in backend.structures(spec):
                raise ValueError("Disallowed structural category")
            if split in ("validation", "test") and category not in observed_train:
                raise ValueError("IID structure absent from training")
            identity = backend.identity(e.observation)
            trace = stable_hash(e.target_trajectory)
            if e.example_id in ids or identity in identities or e.problem in problems or trace in trajectories:
                raise ValueError("Duplicate or cross-split problem/operand/trajectory leakage")
            ids.add(e.example_id); identities.add(identity)
            problems.add(e.problem); trajectories.add(trace)
    return True


def nested_subsets(examples, sizes, seed):
    if any(type(n) is not int or n <= 0 or n > len(examples) for n in sizes):
        raise ValueError("Subset budgets must be positive and available")
    order = list(examples)
    random.Random(seed).shuffle(order)
    return {n: order[:n] for n in sorted(set(sizes))}
