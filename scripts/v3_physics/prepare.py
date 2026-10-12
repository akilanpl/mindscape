"""CPU-only immutable engine dataset preparation; never initializes neural models."""

import argparse
import json
from pathlib import Path

from mindscape.v3.physics_world.data import freeze, load

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--pilot", action="store_true")
    a = p.parse_args()
    if Path(a.output).exists():
        m = load(a.output)
    else:
        m = freeze(
            a.output,
            {"train": 2, "validation": 1, "iid": 1, "ood": 1} if a.pilot else None,
            seed=20261013 if a.pilot else 20261012,
        )
    print(
        json.dumps(
            {
                "dataset": a.output,
                "hash": m["hash"],
                "scope": m["scope"],
                "scenes": len(m["records"]),
            }
        )
    )
