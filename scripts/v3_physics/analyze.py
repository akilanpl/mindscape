import argparse
import json

from mindscape.v3.physics_world.analysis import analyze

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    x = analyze(a.run, a.output)
    print(
        json.dumps({k: x[k] for k in ("primary_completed", "primary_required", "primary_complete")})
    )
