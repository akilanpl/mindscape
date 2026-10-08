import argparse
from mindscape.core.config import load_config
from mindscape.training.run import train


def main():
    parser = argparse.ArgumentParser(description="Train a local task MLP; no model downloads")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--kind", choices=["baseline", "mindscape"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--budget", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    for key in ["kind", "seed", "budget"]:
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    _, metadata = train(args.dataset, config, args.output)
    print({key: metadata[key] for key in ["regime", "parameter_count", "training_rows", "training_time", "checkpoint"]})


if __name__ == "__main__":
    main()
