"""Training-only falsification: literal-blind features cannot rank two different identities."""

import json
from pathlib import Path

import numpy as np
import torch

from mindscape.v3.coding_learning.data import load_public, load_scorer
from mindscape.v3.coding_learning.execution import Executor
from mindscape.v3.coding_learning.memory import VerifiedMemory
from mindscape.v3.coding_learning.planning import ProposalEvaluator, features
from mindscape.v3.coding_learning.schema import RepairProposal

if __name__ == "__main__":
    tasks, _ = load_public("configs/v3_coding/benchmark_v1")
    private = load_scorer("configs/v3_coding/benchmark_v1")
    task = next(t for t in tasks if t.split == "train" and t.family == "accumulator")
    correct = private[task.identity]["teacher_repository"]
    wrong = {path: source.replace("result = 1", "result = 2") for path, source in correct.items()}
    executor = Executor("work/coding/runtime")
    records = json.loads(Path("results/v3_coding/training_v1/memory.json").read_text())
    memory = VerifiedMemory([t for t in tasks if t.split == "train"], records)
    retrieved = [
        r
        for r in memory.retrieve(task.repository, task.contract, k=24)
        if r["task_id"] != task.identity
    ][:2]
    good = features(task.repository, RepairProposal(correct), task.contract, retrieved)
    bad = features(task.repository, RepairProposal(wrong), task.contract, retrieved)
    model = ProposalEvaluator(11)
    model.load_state_dict(
        torch.load("results/v3_coding/training_v1/evaluator-s11.pt", weights_only=True)["state"]
    )
    result = {
        "scope": "post hoc training-only representation falsification; not benchmark augmentation",
        "task_id": task.identity,
        "correct": correct,
        "wrong": wrong,
        "feature_vectors_identical": bool(np.array_equal(good, bad)),
        "correct_visible": executor.run(correct, task.visible_tests),
        "wrong_visible": executor.run(wrong, task.visible_tests),
        "predicted_correct": model.predict(good),
        "predicted_wrong": model.predict(bad),
        "implication": "Current literal-blind evaluator cannot distinguish correct accumulator identity1 from wrong2. Strong fit to simple training negatives does not establish semantic repair evaluation.",
    }
    Path("results/v3_coding/representation-collision.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k
                in (
                    "feature_vectors_identical",
                    "predicted_correct",
                    "predicted_wrong",
                    "implication",
                )
            }
        )
    )
