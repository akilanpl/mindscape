"""Hash-checked entrypoint for the fresh transport-V2 study."""

import argparse
import hashlib
import json
from pathlib import Path

from batched_study import run

from mindscape.v3.coding_learning.agent import CONDITIONS

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--runtime", required=True)
    p.add_argument("--training", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--manifest", default="configs/v3_coding/benchmark_v2")
    p.add_argument("--protocol", default="configs/v3_coding/protocol_v2.json")
    p.add_argument("--cache", default="results/v3_coding/v2-cache")
    p.add_argument("--seconds", type=int, default=1800)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    a = p.parse_args()
    protocol = json.loads(Path(a.protocol).read_text())
    folder = Path(__file__).parent
    for name, key in (
        ("path_schema.py", "transport_source_sha256"),
        ("transport_v2_study.py", "transport_driver_sha256"),
    ):
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != protocol[key]:
            raise ValueError("Frozen V2 transport source changed")
    a.path_schema = True
    run(a)
