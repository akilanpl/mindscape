"""Fixed preliminary protocol: identical budgets/seeds and shared evaluation splits."""
import argparse
from collections import Counter
import copy
from dataclasses import asdict
import json
from pathlib import Path

from mindscape.core.config import load_config
from mindscape.data.generation import load_dataset
from mindscape.data.backends import stable_hash
from mindscape.evaluation.runner import run
from mindscape.models.learned import LearnedModel
from mindscape.training.run import train


def main():
    parser = argparse.ArgumentParser(description='Run preliminary learned comparison; no test-set tuning')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--config', default='configs/experiments/learning.toml')
    parser.add_argument('--output', required=True, help='New directory')
    parser.add_argument('--budgets', type=int, nargs='+', default=[50, 100, 250])
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    config = load_config(args.config)
    splits, manifest = load_dataset(args.dataset)
    fairness = {
        'dataset_hash': stable_hash(manifest), 'same_subset_ids': True,
        'same_test_and_validation_splits': True, 'same_seed_policy': True,
        'same_evaluator': 'mindscape.evaluation.runner.run',
        'shared_backbone': f'52 inputs -> {config["hidden"]} tanh units',
        'optimizer_steps': config['steps'], 'batch_size': config['batch_size'],
        'selection': 'minimum validation cross entropy, no test selection',
        'differences': ['baseline eight digit heads plus sign vs four action classes',
            'policy receives more labels per problem through trajectory supervision',
            'policy can use prescribed exact arithmetic environment; baseline cannot',
            'one legal action means masked success is guaranteed independently of learning',
            'unmasked policy is diagnostic; no instruction to use it as primary superiority result'],
        'subset_ids': manifest['nested_subsets'],
        'seed_schedule': {str(n): args.seeds if n == min(args.budgets) else args.seeds[:1]
                          for n in args.budgets}}
    (output / 'fairness.json').write_text(json.dumps(fairness, indent=2))
    records = []
    for budget in args.budgets:
        seeds = args.seeds if budget == min(args.budgets) else args.seeds[:1]
        for seed in seeds:
            for kind, regime in [('baseline', 'answer_only'), ('mindscape', 'trajectory_supervised')]:
                folder = output / f'{kind}_n{budget}_seed{seed}'
                model, metadata = train(args.dataset, {**config, 'kind': kind, 'seed': seed, 'budget': budget}, folder)
                # Test the on-disk checkpoint rather than relying on trainer object state.
                model = LearnedModel.load(folder / 'checkpoint')
                for split in ['test', 'ood_test']:
                    eval_folder, metrics = run(model, args.dataset, output / 'evaluations', split, regime, budget,
                        notes='PRELIMINARY: prescribed exact arithmetic and singleton mask confound policy success')
                    records.append({'kind': kind, 'mask': kind == 'mindscape', 'training': metadata,
                                    'evaluation_directory': str(eval_folder), 'metrics': metrics})
                    if kind == 'mindscape':
                        diagnostic = copy.copy(model)
                        diagnostic.mask = False
                        diagnostic.identifier += '_unmasked'
                        raw_folder, raw_metrics = run(diagnostic, args.dataset, output / 'evaluations', split, regime, budget,
                            notes='PRELIMINARY unmasked diagnostic: invalid learned actions terminate episodes')
                        records.append({'kind': 'mindscape_unmasked', 'mask': False, 'training': metadata,
                                        'evaluation_directory': str(raw_folder), 'metrics': raw_metrics})
                print(f'{kind}: budget={budget}, seed={seed}, seconds={metadata["training_time"]:.4f}', flush=True)
    (output / 'comparison.json').write_text(json.dumps(records, indent=2))
    raw = []
    for record in records:
        m = record['metrics']
        raw.append({'model': record['kind'], 'budget': m['dataset_size'],
                    'seed': record['training']['config']['seed'], 'split': m['split'],
                    **{key: m[key] for key in ['accuracy', 'grounded_rate', 'trajectory_validity',
                        'goal_success_rate', 'unsupported_rate', 'training_time', 'inference_time',
                        'model_calls', 'parameter_count']},
                    'validation_label_accuracy': record['training']['validation_metrics']['validation_label_accuracy']})
    (output / 'curves.json').write_text(json.dumps(raw, indent=2))
    failures = []
    for record in records:
        rows = [json.loads(line) for line in (Path(record['evaluation_directory']) / 'predictions.jsonl').read_text().splitlines()]
        failures.append({'model': record['kind'], 'budget': record['metrics']['dataset_size'],
                         'seed': record['training']['config']['seed'], 'split': record['metrics']['split'],
                         'error_counts': dict(Counter(row['error_type'] for row in rows))})
    (output / 'failure_analysis.json').write_text(json.dumps(failures, indent=2))
    # Demo is kept separate from all training and validation adapters.
    from mindscape.core.schema import Observation
    model = LearnedModel.load(output / f'mindscape_n{min(args.budgets)}_seed{args.seeds[0]}' / 'checkpoint')
    view = {'environment': 'integer_multiplication', 'problem': '19 × 3',
            'observation': asdict(Observation((19, 3)))}
    prediction = model.predict(view)
    demonstration = copy.deepcopy(model.last_episode)
    model.mask = False
    unmasked = model.predict(view)
    demonstration['unmasked_demo'] = copy.deepcopy(model.last_episode)
    baseline = LearnedModel.load(output / f'baseline_n{min(args.budgets)}_seed{args.seeds[0]}' / 'checkpoint')
    demonstration['baseline_prediction'] = asdict(baseline.predict(view))
    demonstration['trained_policy_prediction'] = asdict(prediction)
    demonstration['unmasked_prediction'] = asdict(unmasked)
    (output / '19_times_3_learned.json').write_text(json.dumps(demonstration, indent=2))
    print(output, flush=True)


if __name__ == '__main__':
    main()
