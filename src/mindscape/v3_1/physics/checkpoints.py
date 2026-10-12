"""Derived-memory artifacts need their own directory before generation writes."""

from pathlib import Path

from mindscape.v3.physics_world.training import checkpoint


def save_derived_memory(directory, model, optimizer, step, identity):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    pointer = directory / "checkpoint-index.json"
    checkpoint(pointer, model, optimizer, step, identity)
    return pointer
