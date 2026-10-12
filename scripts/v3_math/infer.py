"""Run an existing hash-verified compact or residual-adapter checkpoint on a new problem."""

import argparse
import json
from pathlib import Path

import torch

from mindscape.v3.math_learning.data import load
from mindscape.v3.math_learning.environment import Problem, run_problem
from mindscape.v3.math_learning.learner import Predictor, TransitionNet
from mindscape.v3.math_learning.training import build_training, file_hash
from mindscape.v3.schema import canonical


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--manifest", default="configs/v3_math")
    p.add_argument("--operands", type=int, nargs="+", required=True)
    p.add_argument("--operations", nargs="+", required=True)
    p.add_argument(
        "--mode",
        default="concepts",
        choices=("concepts", "procedural", "no_memory", "no_planning", "answer_only"),
    )
    p.add_argument("--pretrained-path")
    p.add_argument("--cache", default="work/math-inference-features.npz")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    root = Path(a.checkpoint_dir)
    receipt = json.loads((root / "training.json").read_text())
    checkpoint = root / "checkpoint.pt"
    if file_hash(checkpoint) != receipt["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    frozen, tasks = load(a.manifest)
    if saved["identity"]["manifest_hash"] != frozen["sha256"]:
        raise ValueError("Manifest mismatch")
    problem = Problem(tuple(a.operands), tuple(a.operations), "unit", "user_inference")
    if problem.identity in frozen["body"]["lockbox_ids"]:
        raise ValueError("Sealed lockbox identity; final protocol required")
    encoder = None
    size = 0
    if saved["identity"]["pretrained_hashes"]:
        if not a.pretrained_path:
            raise ValueError("Approved original local backbone path required")
        from mindscape.v3.math_learning.pretrained import FrozenEncoder

        encoder = FrozenEncoder(a.pretrained_path, a.cache)
        size = encoder.size
        if encoder.hashes != saved["identity"]["pretrained_hashes"]:
            raise ValueError("Pretrained weights mismatch")
    model = TransitionNet(saved["identity"]["seed"], size)
    model.load_state_dict(saved["model"])
    _, memory, _, _ = build_training(tasks["train"][: receipt["ledger"]["distinct_examples"]])
    predictor = Predictor(
        model,
        memory if a.mode in ("concepts", "no_planning") else {},
        a.mode in ("concepts", "no_memory"),
        encoder,
    )
    if a.mode == "answer_only":
        answer, logits = predictor.answer(problem.visible())
        trace = {"answer": str(answer), "answer_logits": logits, "problem": problem.visible()}
    else:
        trace = run_problem(problem, predictor)
    trace["checkpoint_sha256"] = receipt["checkpoint_sha256"]
    trace["condition"] = a.mode
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(canonical(trace) + "\n")
    print(json.dumps({k: trace[k] for k in ("answer", "success") if k in trace}))


if __name__ == "__main__":
    main()
