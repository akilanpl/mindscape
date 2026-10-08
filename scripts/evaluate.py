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
    parser.add_argument("--checkpoint", help="Local trained checkpoint; otherwise deterministic reference")
    parser.add_argument("--unmasked", action="store_true", help="Diagnostic: let learned policy fail on illegal choices")
    args = parser.parse_args()
    model = ReferenceModel()
    if args.checkpoint:
        from mindscape.models.learned import LearnedModel, MindscapeModel
        model = LearnedModel.load(args.checkpoint)
        if args.unmasked:
            if not isinstance(model, MindscapeModel):
                parser.error("--unmasked requires a policy checkpoint")
            model.mask = False
            model.identifier += "_unmasked"
    folder, metrics = run(model, args.dataset, args.output, args.split, args.regime,
                          args.training_size, not args.no_predictions,
                          "Preliminary development run; singleton mask makes policy success non-diagnostic" if args.checkpoint else "Prescribed arithmetic algorithm; no training performed")
    print(folder)
    print({k: metrics[k] for k in ["accuracy", "ood_accuracy", "grounded_rate", "trajectory_validity", "goal_success_rate"]})


if __name__ == "__main__":
    main()
