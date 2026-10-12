"""Post hoc fixed-data controls: retain empirical memory/planning while zeroing weights."""

import argparse
import json
from pathlib import Path

import torch

from mindscape.v3.math_learning.data import load
from mindscape.v3.math_learning.experiment import evaluate
from mindscape.v3.math_learning.learner import Predictor, TransitionNet
from mindscape.v3.math_learning.training import build_training
from mindscape.v3.schema import digest


def run(root, manifest):
    root = Path(root)
    identity = json.loads((root / "identity.json").read_text())
    _, tasks = load(manifest)
    _, memory, ledger, _ = build_training(tasks["train"][:1000])
    torch.set_num_threads(2)
    for condition, seeds in (("zero_full_memory", (0,)), ("random_full_memory", (0, 1, 2))):
        for seed in seeds:
            # Downstream weights zero/random are the intervention; no pretrained encoder
            # needed because zero outputs ignore frozen features. For random control this
            # diagnostic uses the compact backbone, labelled distinctly from adapter study.
            model = TransitionNet(seed)
            if condition.startswith("zero"):
                for parameter in model.parameters():
                    parameter.data.zero_()
            predictor = Predictor(model, memory, True)
            eid = {
                "run_hash": digest(identity),
                "condition": condition,
                "seed": seed,
                "diagnostic": "post_hoc_retained_memory_and_planning_compact_backbone",
                "training_examples": ledger["distinct_examples"],
            }
            rows = evaluate(
                tasks["validation"] + tasks["iid"] + tasks["ood"],
                predictor,
                root / f"control-{condition}-s{seed}",
                condition,
                eid,
            )
            group = [r for r in rows if r["split"] == "ood"]
            print(condition, seed, sum(r["success"] for r in group), len(group))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--manifest", required=True)
    a = p.parse_args()
    run(a.run, a.manifest)
