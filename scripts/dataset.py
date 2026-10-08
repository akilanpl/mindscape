import argparse
from collections import Counter
from mindscape.data.generation import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Validate or inspect a dataset including all splits")
    parser.add_argument("command", choices=["validate", "inspect"])
    parser.add_argument("dataset")
    args = parser.parse_args()
    splits, manifest = load_dataset(args.dataset)
    print("Dataset valid; all checksums, targets, splits and leakage checks passed")
    if args.command == "inspect":
        for split, rows in splits.items():
            print(split, len(rows), dict(Counter(e.metadata["structural_category"] for e in rows)))
        print("Nested budgets:", list(manifest["nested_subsets"]))


if __name__ == "__main__":
    main()
