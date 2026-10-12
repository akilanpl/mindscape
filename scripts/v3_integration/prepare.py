"""Freeze a tiny new request suite; no historical lockbox access."""

from dataclasses import asdict
from pathlib import Path

from mindscape.v3.coding_learning.data import make_task
from mindscape.v3.integration.runtime import Request
from mindscape.v3.physics_world.storage import atomic_json
from mindscape.v3.schema import digest

root = Path("configs/v3_integration")
if (root / "requests.json").exists():
    raise ValueError("Suite already frozen")
requests = [
    Request("mathematics", {"operands": [487, 376], "operations": ["add"]}),
    Request("mathematics", {"operands": [73, 29], "operations": ["sub"]}),
]
task, private = make_task("iid", "threshold", 20261071, 4)
requests.append(
    Request(
        "coding",
        {
            "repository": task.repository,
            "contract": asdict(task.contract),
            "visible_tests": task.visible_tests,
        },
    )
)
requests.append(
    Request(
        "physics",
        {"family": "motion", "seed": 20261071, "index": 1, "horizon": 8, "target": [0.0, 1.5]},
    )
)
atomic_json(
    root / "requests.json",
    {"requests": [asdict(r) for r in requests], "hash": digest([asdict(r) for r in requests])},
)
atomic_json(root / "private.json", {requests[2].identity: private["private_tests"]})
print("Frozen four new integration requests; historical sealed data unused.")
