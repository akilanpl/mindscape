import argparse
from mindscape.evaluation.runner import run
from mindscape.models.reference import ReferenceModel


def main():
    parser = argparse.ArgumentParser(description="Run a deterministic infrastructure reference, not a learned model")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default="results")
    parser.add_argument("--split", choices=["validation", "test", "ood_test"], default="test")
    parser.add_argument("--regime", choices=["answer_only", "trajectory_supervised", "experiential"], default="experiential")
    parser.add_argument("--training-size", type=int)
    parser.add_argument("--no-predictions", action="store_true")
    args = parser.parse_args()
    folder, metrics = run(ReferenceModel(), args.dataset, args.output, args.split, args.regime,
                          args.training_size, not args.no_predictions,
                          "Prescribed arithmetic algorithm; no training performed")
    print(folder)
    print({k: metrics[k] for k in ["accuracy", "ood_accuracy", "grounded_rate", "trajectory_validity", "goal_success_rate"]})


if __name__ == "__main__":
    main()
