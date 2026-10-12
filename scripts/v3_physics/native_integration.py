"""Real MPS checkpoint/optimizer resume equivalence; never runnable on CPU."""

import argparse
import json
from pathlib import Path

from mindscape.v3.physics_world.accelerator import require_mps
from mindscape.v3.physics_world.data import load
from mindscape.v3.physics_world.storage import atomic_json, source_hash
from mindscape.v3.physics_world.training import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    require_mps()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for family in ("motion", "interaction"):
        identity = {
            "family": family,
            "budget": 2,
            "variant": "mindscape",
            "seed": 11,
            "dataset": load(args.dataset)["hash"],
            "source": source_hash(),
            "purpose": "resume-equivalence-pilot",
        }
        _, partial = train(
            args.dataset,
            root / f"{family}/resumed",
            identity,
            updates=12,
            batch_size=16,
            stop_after=6,
        )
        resumed, receipt = train(
            args.dataset, root / f"{family}/resumed", identity, updates=12, batch_size=16
        )
        original, full = train(
            args.dataset, root / f"{family}/uninterrupted", identity, updates=12, batch_size=16
        )
        difference = max(
            float((p - original.state_dict()[name]).abs().max().item())
            for name, p in resumed.state_dict().items()
        )
        if difference > 1e-5:
            raise RuntimeError(f"Resume differs:{difference}")
        result = {
            "family": family,
            "max_weight_difference": difference,
            "resumed_updates": receipt["new_updates"],
            "full_updates": full["new_updates"],
            "device": "mps",
            "optimizer_state_resumed": True,
        }
        results.append(result)
        del original, resumed
    atomic_json(
        root / "resume-equivalence.json",
        {"passed": True, "results": results, "tolerance": 1e-5, "neural_cpu_execution": False},
    )
    print(json.dumps(results, indent=2))
