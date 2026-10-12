"""Freeze fresh V2 holdouts using the unchanged V1 generator and training pool."""

import json
import random
import re
from dataclasses import asdict
from pathlib import Path

from mindscape.v3.coding_learning.data import (
    OOD_FAMILIES,
    TRAIN_FAMILIES,
    load_public,
    load_scorer,
    make_task,
)
from mindscape.v3.coding_learning.execution import Executor
from mindscape.v3.schema import digest

if __name__ == "__main__":
    root = Path("configs/v3_coding/benchmark_v2")
    if root.exists():
        raise ValueError("V2 benchmark already frozen")
    original, old_manifest = load_public("configs/v3_coding/benchmark_v1")
    old_private = load_scorer("configs/v3_coding/benchmark_v1")
    old_receipts = json.loads(Path("configs/v3_coding/benchmark_v1/validation.json").read_text())
    reused = [t for t in original if t.split in ("train", "validation")]
    public = [asdict(t) for t in reused]
    private = [old_private[t.identity] for t in reused]
    ids = {t.identity for t in reused}
    receipts = [r for r in old_receipts if r["task_id"] in ids]
    used = {int(re.search(r"\d+", t.contract.description).group()) for t in original}
    rng = random.Random(840017)
    ks = iter(rng.sample(sorted(set(range(2, 180)) - used), 18))
    executor = Executor("work/coding/runtime")
    for split, families, repeats in (("iid", TRAIN_FAMILIES, 1), ("ood", OOD_FAMILIES, 2)):
        for family in families:
            for _ in range(repeats):
                task, secret = make_task(split, family, rng.randrange(10**9), next(ks))
                outcomes = {}
                if task.identity in old_manifest["task_ids"]:
                    raise ValueError("Old task reused")
                for label, repo in (
                    ("broken", task.repository),
                    ("teacher", secret["teacher_repository"]),
                ):
                    for name, suite in (
                        ("visible", task.visible_tests),
                        ("private", secret["private_tests"]),
                    ):
                        outcomes[label + "_" + name] = executor.run(repo, suite)
                if any(
                    outcomes[key]["all_passed"] for key in ("broken_visible", "broken_private")
                ) or not all(
                    outcomes[key]["all_passed"] for key in ("teacher_visible", "teacher_private")
                ):
                    raise ValueError("Executed V2 task validation failed")
                public.append(asdict(task))
                private.append(secret)
                receipts.append({"task_id": task.identity, "executions": outcomes})
    pub = {"schema": "mindscape.v3.coding/1", "tasks": public}
    sec = {"schema": "mindscape.v3.coding.scorer/1", "tasks": private}
    manifest = {
        "public_hash": digest(pub),
        "private_hash": digest(sec),
        "task_ids": [t["identity"] for t in public],
        "counts": {"train": 24, "validation": 6, "iid": 6, "ood": 12},
        "validation_hash": digest(receipts),
        "seed": 840017,
        "reference_policy": "unchanged24 approved training examples; fresh18 holdouts; terminal private scoring",
        "reused_executed_train_validation_tasks": 30,
        "fresh_executed_holdout_tasks": 18,
        "excluded_v1_holdout_ids": [t.identity for t in original if t.split in ("iid", "ood")],
    }
    root.mkdir(parents=True)
    for name, value in (
        ("public.json", pub),
        ("private.json", sec),
        ("manifest.json", manifest),
        ("validation.json", receipts),
    ):
        (root / name).write_text(json.dumps(value, indent=2) + "\n")
    protocol = json.loads(Path("configs/v3_coding/protocol.json").read_text())
    ood = [t for t in public if t["split"] == "ood"]
    iid = [t for t in public if t["split"] == "iid"]
    protocol.update(
        schema="mindscape.v3.coding.protocol/2",
        seed=840017,
        evaluation_task_ids=[t["identity"] for t in ood[::2] + iid + ood[1::2]],
        transport="explicit allowed paths, no fictitious filename example",
        training_pool="exactly V1 training IDs/checkpoints; no added examples or Qwen updates",
        prior_exposure="V1 transport failures observed. V2 holdouts fresh and frozen before any V2 model call.",
        target_unchanged=True,
        generator_unchanged=True,
    )
    Path("configs/v3_coding/protocol_v2.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps(manifest))
