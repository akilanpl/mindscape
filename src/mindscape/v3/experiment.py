"""Frozen manifests, explicit leakage checks, atomic resume and raw replayable traces."""

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from .domains import DOMAINS, Task, generate_tasks
from .kernel import Kernel, replay
from .models import TinyMLP
from .schema import canonical, digest

SPLITS = {"train", "validation", "iid", "ood"}


def validate_tasks(tasks):
    identities = set()
    for task in tasks:
        if task.split not in SPLITS or task.domain not in DOMAINS:
            raise ValueError("Unknown split/domain")
        if task.identity in identities:
            raise ValueError("Duplicate identity / cross-partition leakage")
        identities.add(task.identity)
    for domain in DOMAINS:
        if {t.split for t in tasks if t.domain == domain} != SPLITS:
            raise ValueError("Every domain requires all four frozen splits")


def freeze_manifest(path):
    tasks = generate_tasks()
    validate_tasks(tasks)
    body = {
        "schema": "mindscape.v3.manifest/1",
        "seed": 20261012,
        "tasks": [asdict(t) for t in tasks],
        "policy": "all_65_syntax_candidates_no_semantic_mask",
        "scope": "small_CPU_foundation_smoke_not_domain_training",
    }
    manifest = {"body": body, "sha256": digest(body)}
    atomic_json(Path(path), manifest, exclusive=True)
    return manifest


def read_manifest(path):
    value = json.loads(Path(path).read_text())
    if value["sha256"] != digest(value["body"]):
        raise ValueError("Frozen manifest checksum mismatch")
    if value["body"]["schema"] != "mindscape.v3.manifest/1":
        raise ValueError("Unsupported manifest")
    tasks = tuple(
        Task(t["domain"], tuple(t["inputs"]), t["split"], t["family"])
        for t in value["body"]["tasks"]
    )
    validate_tasks(tasks)
    return value, tasks


def atomic_json(path, value, exclusive=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open("x") as file:
            file.write(canonical(value) + "\n")
            file.flush()
            os.fsync(file.fileno())
        return
    tmp = path.with_suffix(".tmp")
    with tmp.open("x") as file:
        file.write(canonical(value) + "\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(tmp, path)


@contextmanager
def lock(directory):
    path = directory / ".lock"
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        yield
    finally:
        os.close(fd)
        path.unlink()


def source_hash():
    root = Path(__file__).parent
    return digest(
        {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("*.py"))}
    )


def wilson(successes, total):
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Invalid binomial counts")
    z = 1.959963984540054
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [max(0, center - radius), min(1, center + radius)]


def run(manifest_path, output, seeds=(0, 1, 2), conditions=("random", "zero")):
    manifest, tasks = read_manifest(manifest_path)
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or not conditions
        or len(set(conditions)) != len(conditions)
    ):
        raise ValueError("Unique seeds and conditions required")
    if any(c not in ("random", "zero") for c in conditions):
        raise ValueError("Unknown condition")
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    source = source_hash()
    models = {
        (condition, seed): TinyMLP(seed, condition == "zero")
        for condition in conditions
        for seed in seeds
    }
    identity = {
        "manifest_hash": manifest["sha256"],
        "source_hash": source,
        "seeds": list(seeds),
        "conditions": list(conditions),
        "checkpoints": {f"{c}:{s}": digest(m.checkpoint()) for (c, s), m in models.items()},
        "runtime": {"python": sys.version, "platform": platform.platform()},
    }
    resumed = 0
    with lock(directory):
        receipt = directory / "identity.json"
        if receipt.exists():
            if json.loads(receipt.read_text()) != identity:
                raise ValueError("Resume identity mismatch")
        else:
            atomic_json(receipt, identity, exclusive=True)
        rows = []
        for (condition, seed), model in models.items():
            kernel = Kernel(model)
            checkpoint = directory / f"checkpoint-{condition}-{seed}.json"
            if checkpoint.exists():
                TinyMLP.load(checkpoint, digest(model.checkpoint()))
            else:
                model.save(checkpoint)
            for task in tasks:
                # No training is performed in smoke controls; training records are not outcomes.
                if task.split == "train":
                    continue
                key = f"{condition}-{seed}-{task.identity}"
                path = directory / "traces" / f"{key}.json"
                if path.exists():
                    saved = json.loads(path.read_text())
                    trace = saved["trace"]
                    if saved["sha256"] != digest(trace):
                        raise ValueError("Corrupt completed trace")
                    if trace["provenance"] != {
                        "task_id": task.identity,
                        "split": task.split,
                        "model_hash": digest(model.checkpoint()),
                        "source_hash": source,
                        "seed": seed,
                    }:
                        raise ValueError("Trace provenance mismatch")
                    replay(trace, DOMAINS[task.domain])
                    resumed += 1
                else:
                    trace = kernel.run(task, DOMAINS[task.domain], seed, source)
                    replay(trace, DOMAINS[task.domain])
                    atomic_json(path, {"sha256": digest(trace), "trace": trace}, exclusive=True)
                rows.append(
                    {
                        "key": key,
                        "condition": condition,
                        "seed": seed,
                        "domain": task.domain,
                        "split": task.split,
                        "task_id": task.identity,
                        "trace_hash": digest(trace),
                        "success": trace["success"],
                        "latency_seconds": trace["latency_seconds"],
                        "cpu_seconds": trace["cpu_seconds"],
                    }
                )
        cells = []
        for condition in conditions:
            for seed in seeds:
                for domain in DOMAINS:
                    for split in ("validation", "iid", "ood"):
                        group = [
                            r
                            for r in rows
                            if (r["condition"], r["seed"], r["domain"], r["split"])
                            == (condition, seed, domain, split)
                        ]
                        n, k = len(group), sum(r["success"] for r in group)
                        cells.append(
                            {
                                "condition": condition,
                                "seed": seed,
                                "domain": domain,
                                "split": split,
                                "successes": k,
                                "total": n,
                                "wilson_95_task_interval": wilson(k, n),
                            }
                        )
        summary = {
            "identity_hash": digest(identity),
            "rows": rows,
            "cells": cells,
            "completed": len(rows),
            "resumed": resumed,
            "latency_seconds": sum(r["latency_seconds"] for r in rows),
            "cpu_seconds": sum(r["cpu_seconds"] for r in rows),
            "statistical_unit": "unique tasks per seed; seeds are not pooled",
            "paid_cost_usd": 0,
            "weights_trained": False,
        }
        atomic_json(directory / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output")
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        print(freeze_manifest(args.manifest)["sha256"])
    else:
        if not args.output:
            parser.error("--output required for execution")
        result = run(args.manifest, args.output)
        print(canonical({k: result[k] for k in ("completed", "resumed", "cpu_seconds")}))


if __name__ == "__main__":
    main()
