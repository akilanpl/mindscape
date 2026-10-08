import argparse
from collections import Counter
from mindscape.data.generation import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Validate or inspect a dataset including all splits")
    parser.add_argument("command", choices=["validate", "inspect", "export"])
    parser.add_argument("dataset")
    parser.add_argument("--form", choices=["answer_only", "trajectory_supervised", "experiential"], default="answer_only")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", help="New JSONL file for export")
    args = parser.parse_args()
    splits, manifest = load_dataset(args.dataset)
    if args.command != "export":
        print("Dataset valid; all checksums, targets, splits and leakage checks passed")
    if args.command == "export":
        import json
        if not args.output:
            parser.error("export requires --output")
        with open(args.output, "x") as stream:
            for example in splits[args.split]:
                stream.write(json.dumps(example.view(args.form, supervision=args.split == "train")) + "\n")
    if args.command == "inspect":
        for split, rows in splits.items():
            print(split, len(rows), dict(Counter(e.metadata["structural_category"] for e in rows)))
        print("Nested budgets:", list(manifest["nested_subsets"]))


if __name__ == "__main__":
    main()
