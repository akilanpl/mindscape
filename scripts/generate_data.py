import argparse
from pathlib import Path
from mindscape.core.config import load_config
from mindscape.data.generation import generate, save_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate verified, identity-disjoint JSONL datasets")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory; never overwritten")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    splits = generate(config)
    save_dataset(args.output, splits, config)
    print({split: len(rows) for split, rows in splits.items()})


if __name__ == "__main__":
    main()
