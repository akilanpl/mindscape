"""Pure evidence fixtures. No model, torch, or optimizer invocation."""
import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / 'cloud/kaggle/scripts/comparison_report.py'
spec = importlib.util.spec_from_file_location('comparison_report', PATH)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def records(seed):
    return [dict(normalized_identity=f'{seed}-m', domain='math', prediction={'math_digit': 1, 'math_carry': 0}, reference={'math_digit': 1, 'math_carry': 0}, repaired=False),
            dict(normalized_identity=f'{seed}-c', domain='coding', prediction={'coding_index': 0}, reference={'coding_index': 0}, repaired=False),
            dict(normalized_identity=f'{seed}-p', domain='physics', prediction={'physics_delta': [1., 2.]}, reference={'physics_delta': [2., 2.]}, repaired=False)]


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    fits, evaluation, historical, out = [tmp_path / p for p in ('fits', 'evaluation', 'historical', 'out')]
    data_files, summary, jobs = {}, [], []
    for seed in report.SEEDS:
        data = historical / f'data-v1/evaluator-only/final-{seed}.json'
        write(data, {'seed': seed, 'pure_fixture': True})
        digest = report.sha(data)
        data_files[f'evaluator-only/final-{seed}.json'] = digest
        recs = records(seed); m = report.metrics(recs)
        runs = []
        for condition in report.CONDITIONS:
            fit = {'seed': seed, 'condition': condition, 'checkpoint_sha256': f'{seed}-{condition}',
                   'training_sha256': str(seed), 'optimizer_updates': 2000, 'torch': 'fixture-cloud'}
            write(fits / f'{seed}-{condition}/result.json', fit)
            value = {'records': recs, 'metrics': m, 'checkpoint_sha256': fit['checkpoint_sha256'], 'perturbations': []}
            path = evaluation / f'{seed}-{condition}.json'; write(path, value)
            runs.append({'seed': seed, 'condition': condition, 'version': 'R57c', 'metrics': m,
                         'artifact': path.name, 'sha256': report.sha(path)})
            for version in ('R56', 'R57', 'R57c'):
                write(historical / f'scoreboard-final/predictions/{seed}-{condition}-{version}-final.json', value)
                summary.append({'seed': seed, 'condition': condition, 'version': version, 'phase': 'final', 'metrics': m})
        jobs.append({'seed': seed, 'evaluation_sha256': digest, 'training_sha256': str(seed),
                     'overlap_audit': {'normalized_overlap': 0, 'global_training_union': {'normalized_overlap': 0}},
                     'runs': runs, 'baselines': {key: {'records': recs, 'metrics': m} for key in ('analytic_rule', 'training_majority_constant')}})
    write(historical / 'data-v1/manifest.json', {'files': data_files})
    write(historical / 'scoreboard-final/scoreboard.json', {'summary': summary, 'fit_receipts': [{'torch': 'fixture-historical'}]})
    write(evaluation / 'evaluation_receipt.json', {'status': 'COMPLETED_CLOUD_EVALUATION', 'device': 'cpu',
          'evaluation_optimizer_updates': 0, 'qualification': False, 'selection_performed': False, 'results': jobs})
    # Unit-test only: integration calls the production verifier unchanged.
    monkeypatch.setattr(report, 'verify_export', lambda path: report.read(path / 'result.json'))
    return dict(fits=fits, evaluation=evaluation, historical=historical, output=out)


def test_complete_verified_fixture_generates_descriptive_report(evidence):
    result = report.build(**evidence)
    assert result['optimizer_updates_verified'] == 18000
    assert len(result['per_seed']) == 36
    assert len(result['seed_descriptions']) == 12
    text = (evidence['output'] / 'CLOUD_COMPARISON.md').read_text()
    assert 'cannot be attributed solely to CUDA' in text
    assert 'non-neural controls' in text
    with pytest.raises(FileExistsError):
        report.build(**evidence)


def test_missing_fit_rejected(evidence):
    next(evidence['fits'].rglob('result.json')).unlink()
    with pytest.raises(ValueError, match='nine exported'):
        report.build(**evidence)
    assert not evidence['output'].exists()


def test_raw_output_tamper_rejected(evidence):
    path = next(p for p in evidence['evaluation'].glob('*.json') if p.name != 'evaluation_receipt.json')
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='raw artifact hash'):
        report.build(**evidence)


def test_wrong_distribution_rejected(evidence):
    path = evidence['evaluation'] / 'evaluation_receipt.json'
    receipt = report.read(path); receipt['results'][0]['evaluation_sha256'] = 'wrong'
    write(path, receipt)
    with pytest.raises(ValueError, match='distribution bytes'):
        report.build(**evidence)


def test_missing_ninth_evaluation_rejected(evidence):
    path = evidence['evaluation'] / 'evaluation_receipt.json'
    receipt = report.read(path); receipt['results'][-1]['runs'].pop()
    write(path, receipt)
    with pytest.raises(ValueError, match='nine independent'):
        report.build(**evidence)


def test_metrics_recomputed_from_predictions():
    actual = report.metrics(records(17))
    assert actual['math']['joint_accuracy'] == 1
    assert actual['physics']['mse'] == .5
    declared = json.loads(json.dumps(actual)); declared['physics']['mse'] = 0
    with pytest.raises(ValueError, match='preserved predictions'):
        report.verify_metrics(declared, actual)


def test_probe_summary_separates_reference_accuracy_from_change():
    value = report.probe_summary([{'problem': {'domain': 'coding'}, 'probe': 'candidate_rotation_1',
              'prediction': {'coding_index': 1}, 'original_prediction': {'coding_index': 0},
              'analytic_reference': {'coding_index': 2}, 'same_selected_name': True}])[0]
    assert value['prediction_changed_fraction'] == 1
    assert value['exact_accuracy_to_analytic_reference'] == 0
    assert value['same_selected_name_fraction'] == 1
