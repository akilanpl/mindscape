"""Run and display a complete, independently checked episode."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from mindscape import __version__
from mindscape.environments.multiplication.environment import MultiplicationEnvironment
from mindscape.verification.verifier import verify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=int)
    parser.add_argument("right", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    env = MultiplicationEnvironment()
    env.reset((args.left, args.right))
    trajectory = env.run()
    verification = verify(trajectory, env.get_state().answer)
    payload = {
        "schema_version": __version__,
        "trajectory": asdict(trajectory),
        "final_state": asdict(env.get_state()),
        "answer": env.get_state().answer,
        "verification": {**asdict(verification), "verified": verification.verified},
        "goal_status": env.is_goal_reached(),
    }
    text = json.dumps(payload, indent=2)
    if args.output:
        with args.output.open("x") as stream:
            stream.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
