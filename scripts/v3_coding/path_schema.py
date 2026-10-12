"""Post-freeze transport remediation. Validate on training tasks only, not V1 holdouts."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch

from mindscape.v3.coding_learning.agent import solve
from mindscape.v3.coding_learning.data import load_public, load_scorer
from mindscape.v3.coding_learning.execution import Executor
from mindscape.v3.coding_learning.experiment import QwenBackend
from mindscape.v3.coding_learning.memory import VerifiedMemory
from mindscape.v3.coding_learning.planning import ProposalEvaluator


class PathSchemaBackend:
    """Change instruction schema only; never rewrite or supply proposed code."""

    def __init__(self, backend):
        self.backend = backend

    def generate(self, payload, max_tokens):
        payload = copy.deepcopy(payload)
        paths = sorted(payload["repository"])
        payload["allowed_edit_paths"] = paths
        payload["instruction"] = (
            "Return only a JSON object with fields hypothesis (brief string) and candidates (list of one or two repairs). "
            "Each repair has exactly one field repository, a mapping from existing paths to their complete corrected Python source. "
            "Choose file paths from allowed_edit_paths exactly; no placeholder or new filenames. "
            "Only files requiring changes need to be included. Do not output Markdown or extra text. "
            "Use the supplied actual visible execution feedback; private tests are inaccessible."
        )
        response, timing = self.backend.generate(payload, max_tokens)
        return response, dict(
            timing, actual_model_payload=payload, transport_schema="explicit-path-v2"
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--runtime", required=True)
    p.add_argument("--training", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--families",
        nargs="+",
        default=["threshold", "exclusive", "mutation", "ordering", "accumulator", "empty"],
    )
    a = p.parse_args()
    root = Path(a.output)
    if root.exists():
        raise ValueError("Diagnostic destination must be new")
    root.mkdir(parents=True)
    tasks, manifest = load_public("configs/v3_coding/benchmark_v1")
    private = load_scorer("configs/v3_coding/benchmark_v1")
    training = [t for t in tasks if t.split == "train"]
    selected = [next(t for t in training if t.family == f) for f in a.families]
    executor = Executor(a.runtime)
    memory = VerifiedMemory.load(Path(a.training) / "memory.json", training, executor, private)
    checkpoint = Path(a.training) / "evaluator-s11.pt"
    ledger = json.loads((Path(a.training) / "ledger.json").read_text())
    if (
        hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        != ledger["receipts"][0]["checkpoint_sha256"]
    ):
        raise ValueError("Checkpoint changed")
    state = torch.load(checkpoint, weights_only=True)
    model = ProposalEvaluator(11)
    model.load_state_dict(state["state"])
    backend = PathSchemaBackend(QwenBackend(a.model, root / "cache"))
    for task in selected:
        trace = solve(task, backend, executor, memory, model, "mindscape")
        trace["diagnostic_scope"] = (
            "post hoc training-only path-schema remediation; no structural-OOD result"
        )
        trace["training_validation_after_termination"] = executor.run(
            trace["repository"], private[task.identity]["private_tests"]
        )
        (root / (task.identity + ".json")).write_text(json.dumps(trace, indent=2))
        print(
            json.dumps(
                {
                    "family": task.family,
                    "training_visible_pass": any(
                        x.get("actual_visible_result", {}).get("all_passed")
                        for x in trace["attempts"]
                    ),
                    "training_private_pass": trace["training_validation_after_termination"][
                        "all_passed"
                    ],
                }
            ),
            flush=True,
        )
