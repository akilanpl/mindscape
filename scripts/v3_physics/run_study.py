"""Native MPS study entrypoint. No CPU-device option exists."""

import argparse

from mindscape.v3.physics_world.study import run

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--pilot", action="store_true")
    a = p.parse_args()
    run(a.dataset, a.output, updates=12 if a.pilot else 400, batch_size=a.batch_size, pilot=a.pilot)
