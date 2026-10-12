"""Load only existing pinned assets. All neural models execute on native MPS."""

import importlib.util
import json
from pathlib import Path

import torch

from mindscape.v3.physics_world.accelerator import require_mps, verify_placement
from mindscape.v3.physics_world.storage import file_sha
from mindscape.v3.schema import digest

from .adapters import Coding, Math, Physics

PYTHON = "/Users/akilan/Documents/Codex/2026-10-08/files-pasted-by-the-user-mindscape/outputs/mindscape/work/final-venv/bin/python"
QWEN = Path(
    "/Users/akilan/Documents/Codex/2026-10-08/files-pasted-by-the-user-mindscape/outputs/mindscape/work/coding/hf/models--Qwen--Qwen2.5-Coder-1.5B-Instruct/snapshots/2e1fd397ee46e1388853d2af2c993145b0f1098a"
)
MATH = Path("work/math-evidence-restored/compact/model-procedural-n1000-s0")
PHYSICS = Path("results/v3_1_physics/native-pilot-1791816435786858000")
TRAINING = Path("results/v3_coding/training_v1")
WASI = Path("work/coding/runtime")


def preflight():
    needed = [
        MATH / "checkpoint.pt",
        MATH / "training.json",
        TRAINING / "evaluator-s11.pt",
        TRAINING / "memory.json",
        WASI / "python.wasm",
        WASI / "python.cwasm",
        QWEN / "model.safetensors",
        PHYSICS / "motion/s11/candidate/checkpoint-index.json",
    ]
    missing = [str(p) for p in needed if not p.is_file()]
    if missing:
        raise FileNotFoundError("Existing assets unavailable: " + ", ".join(missing))
    return {str(p): file_sha(p) for p in needed}


class Assets:
    def __init__(self, private):
        require_mps()
        self.hashes = preflight()
        self.private = private
        self.loaded = {}

    def math(self, request):
        if "math" not in self.loaded:
            from mindscape.v3.math_learning.learner import TransitionNet

            saved = torch.load(MATH / "checkpoint.pt", map_location="mps", weights_only=True)
            receipt = json.loads((MATH / "training.json").read_text())
            if receipt["checkpoint_sha256"] != self.hashes[str(MATH / "checkpoint.pt")]:
                raise ValueError("Math checkpoint hash")
            manifest = json.loads(Path("configs/v3_math/manifest.json").read_text())
            if saved["identity"]["manifest_hash"] != manifest["sha256"]:
                raise ValueError("Math manifest provenance")
            with torch.device("mps"):
                model = TransitionNet(saved["identity"]["seed"])
            model.load_state_dict(saved["model"])
            model.eval()
            verify_placement(model)
            self.loaded["math"] = model
        return Math(
            request,
            self.loaded["math"],
            {
                "checkpoint": self.hashes[str(MATH / "checkpoint.pt")],
                "device": "mps",
                "parameter_counts": {
                    "transition_model": sum(p.numel() for p in self.loaded["math"].parameters())
                },
            },
        )

    def physics(self, request):
        family = request.inputs["family"]
        if family not in self.loaded:
            from mindscape.v3_1.physics.model import Dynamics

            root = PHYSICS / family / "s11/candidate"
            index = json.loads((root / "checkpoint-index.json").read_text())
            if file_sha(root / index["file"]) != index["sha256"]:
                raise ValueError("Physics checkpoint hash")
            saved = torch.load(root / index["file"], map_location="mps", weights_only=True)
            if saved["identity"] != index["identity"]:
                raise ValueError("Physics checkpoint provenance")
            model = Dynamics("candidate", 11)
            model.load_state_dict(saved["state"])
            model.eval()
            verify_placement(model)
            self.loaded[family] = model, index["sha256"]
        model, sha = self.loaded[family]
        return Physics(
            request,
            model,
            {
                "checkpoint": sha,
                "device": "mps",
                "historical_scientific_status": "NO-GO",
                "parameter_counts": {"dynamics": sum(p.numel() for p in model.parameters())},
            },
        )

    def coding(self, request):
        if "coding" not in self.loaded:
            from mindscape.coding.model import LocalCoder
            from mindscape.v3.coding_learning.data import load_public, load_scorer
            from mindscape.v3.coding_learning.execution import Executor
            from mindscape.v3.coding_learning.memory import VerifiedMemory
            from mindscape.v3.coding_learning.planning import ProposalEvaluator

            class MpsEvaluator(ProposalEvaluator):
                def predict(self, x):
                    with torch.inference_mode():
                        z = self(torch.as_tensor(x, device="mps", dtype=torch.float32))
                        return {
                            "visible_usefulness": float(torch.sigmoid(z[0]).item()),
                            "seconds": float(torch.expm1(z[1].clamp(0, 4)).item()),
                            "kind": "hypothetical_unverified",
                            "private_success": None,
                        }

            with torch.device("mps"):
                evaluator = MpsEvaluator(11)
            saved = torch.load(TRAINING / "evaluator-s11.pt", map_location="mps", weights_only=True)
            evaluator.load_state_dict(saved["state"])
            evaluator.eval()
            verify_placement(evaluator)
            executor = Executor(WASI)
            tasks, _ = load_public("configs/v3_coding/benchmark_v1")
            train = [t for t in tasks if t.split == "train"]
            memory = VerifiedMemory.load(
                TRAINING / "memory.json",
                train,
                executor,
                load_scorer("configs/v3_coding/benchmark_v1"),
            )
            coder = LocalCoder(QWEN, cache_enabled=False, device="mps", precision="float16")

            class Generator:
                def generate(self, payload, max_tokens):
                    text = coder.generate(
                        "Return the requested Python repair JSON using only visible evidence.",
                        json.dumps(payload, sort_keys=True),
                        max_tokens,
                        11,
                    )
                    return text, dict(coder.timings[-1])

            spec = importlib.util.spec_from_file_location(
                "preserved_path_schema", "scripts/v3_coding/path_schema.py"
            )
            transport = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(transport)
            backend = transport.PathSchemaBackend(Generator())
            backend.parameter_counts = {
                "qwen": coder.parameter_count,
                "proposal_evaluator": sum(p.numel() for p in evaluator.parameters()),
            }
            self.loaded["coding"] = backend, evaluator, memory, executor
        backend, evaluator, memory, executor = self.loaded["coding"]
        return Coding(
            request,
            backend,
            evaluator,
            memory,
            executor,
            self.private[request.identity],
            {
                "backbone": self.hashes[str(QWEN / "model.safetensors")],
                "evaluator": self.hashes[str(TRAINING / "evaluator-s11.pt")],
                "memory": digest(memory.records),
                "device": "mps",
                "execution": "WASI",
                "parameter_counts": dict(backend.parameter_counts),
            },
        )

    def factories(self):
        return {"mathematics": self.math, "coding": self.coding, "physics": self.physics}
