"""Inference-only R6 scoring. No optimizer, fitting, or checkpoint selection.

Protected final cases are supplied only after the controller locks all fits.
Filesystem/runtime isolation is enforced by the cloud launcher, not asserted
merely because this worker was invoked.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import errno
import os
from pathlib import Path
import sys
import time


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n')


def _state(prediction):
    value = prediction.get('state')
    if (not isinstance(value, (list, tuple)) or len(value) != 4 or
            any(type(x) not in (int, float) or not math.isfinite(x) for x in value)):
        return None
    return list(value)


def _metric(rows, predictions):
    if not rows:
        return {'count': 0}
    if rows[0]['domain'] != 'physics':
        correct = sum(isinstance(p.get('answer'), str) and p['answer'] == str(r['target']['answer'])
                      for r, p in zip(rows, predictions))
        return {'count': len(rows), 'correct': correct, 'exact_accuracy': correct/len(rows),
                'invalid_prediction_count': sum(not isinstance(p.get('answer'), str) for p in predictions)}
    squares, absolutes = [0.]*4, [0.]*4
    valid = 0
    for row, prediction in zip(rows, predictions):
        state = _state(prediction)
        if state is None:
            continue
        target = row['target']['state']
        if len(target) != 4 or any(not math.isfinite(v) for v in target):
            raise ValueError('invalid reference state')
        valid += 1
        for axis in range(4):
            delta = state[axis]-target[axis]
            squares[axis] += delta*delta
            absolutes[axis] += abs(delta)
    finite = {'mse': sum(squares)/(4*valid), 'mae': sum(absolutes)/(4*valid),
              'mse_per_dimension': [x/valid for x in squares],
              'mae_per_dimension': [x/valid for x in absolutes]} if valid else None
    # Never make a model look better by silently omitting invalid outputs.
    return {'count': len(rows), 'finite_prediction_count': valid,
            'invalid_prediction_count': len(rows)-valid,
            'mse': finite['mse'] if finite and valid == len(rows) else None,
            'mae': finite['mae'] if finite and valid == len(rows) else None,
            'mse_per_dimension': finite['mse_per_dimension'] if finite and valid == len(rows) else None,
            'mae_per_dimension': finite['mae_per_dimension'] if finite and valid == len(rows) else None,
            'finite_only_diagnostics': finite,
            'dimension_order': ['vx', 'vy', 'x', 'y'],
            'units': ['m/s', 'm/s', 'm', 'm'],
            'mixed_units_aggregate': 'descriptive only; per-dimension errors are primary'}


def score(rows, predictions):
    """Pure metrics: exact strings, physical errors, split/topic stratification."""
    if len(rows) != len(predictions):
        raise ValueError('prediction count mismatch')
    if len({r['record_id'] for r in rows}) != len(rows):
        raise ValueError('duplicate record identities')
    groups = {'domains': defaultdict(list), 'per_topic': defaultdict(list),
              'per_split': defaultdict(list), 'math_carry_borrow': defaultdict(list)}
    for row, prediction in zip(rows, predictions):
        if not isinstance(prediction, dict):
            raise ValueError('prediction must be an object')
        for group, key in [('domains', row['domain']), ('per_topic', row['domain']+'/'+row['topic']),
                           ('per_split', row['split']+'/'+row['domain'])]:
            groups[group][key].append((row, prediction))
        if row['domain'] == 'mathematics':
            from research.r6.datasets.mathematics import carry_borrow_category
            groups['math_carry_borrow'][row['split']+'/'+carry_borrow_category(row)].append((row, prediction))
    return {'count': len(rows), **{group: {key: _metric([x[0] for x in pairs], [x[1] for x in pairs])
              for key, pairs in values.items()} for group, values in groups.items()}}


def baseline_predictions(rows, normalization):
    """Frozen TRAIN-derived baselines; never estimate a prior on final labels."""
    base = normalization.get('baselines', {})
    majority = base.get('answer_majority', normalization.get('training_majority_answers', normalization.get('answer_majority', {})))
    mean = base.get('physics_mean_state', normalization.get('physics_mean_state'))
    result = []
    for row in rows:
        if row['domain'] == 'physics':
            if mean is None or _state({'state': mean}) is None:
                raise ValueError('training-derived physics mean baseline missing')
            result.append({'state': list(mean)})
        else:
            if row['domain'] not in majority:
                raise ValueError('training-derived majority baseline missing: '+row['domain'])
            result.append({'answer': str(majority[row['domain']])})
    return result


def reference_predictions(rows):
    """External exact engines; separate category, never neural outputs."""
    from research.r6.datasets import coding, mathematics, physics
    result = []
    for row in rows:
        if row['domain'] == 'coding':
            result.append({'answer': str(coding.interpret(row['input']))})
        elif row['domain'] == 'mathematics':
            result.append({'answer': str(mathematics.reference_answer(row['structured_input']['expression']))})
        elif row['domain'] == 'physics':
            result.append({'state': physics.verify(row['structured_input'])})
        else:
            raise ValueError('unknown domain')
    return result


def make_probes(rows):
    from research.r6.datasets import coding, physics
    result = []
    for index, row in enumerate(rows):
        if row['domain'] not in ('coding', 'physics'):
            continue
        variants = coding.perturbations(row) if row['domain'] == 'coding' else physics.perturbations(row)
        for name, variant in variants.items():
            variant = dict(variant)
            variant['record_id'] = row['record_id']+'-probe-'+name
            result.append({'base_index': index, 'name': name, 'row': variant})
    return result


def score_probes(rows, original_predictions, probes, probe_predictions):
    if len(probes) != len(probe_predictions) or len(rows) != len(original_predictions):
        raise ValueError('probe prediction count mismatch')
    grouped = defaultdict(list)
    for probe, prediction in zip(probes, probe_predictions):
        base = rows[probe['base_index']]
        original = original_predictions[probe['base_index']]
        name = base['split']+'/'+base['domain']+'/'+probe['name']
        detail = {'base_record_id': base['record_id'], 'record_id': probe['row']['record_id'],
                  'reference': probe['row']['target'], 'prediction': prediction}
        if base['domain'] == 'coding':
            detail['correct'] = prediction.get('answer') == probe['row']['target']['answer']
            detail['prediction_unchanged'] = prediction.get('answer') == original.get('answer')
            detail['reference_unchanged'] = probe['row']['target']['answer'] == base['target']['answer']
        else:
            first, second = _state(original), _state(prediction)
            delta = [b-a for a,b in zip(base['target']['state'], probe['row']['target']['state'])]
            detail['reference_response'] = delta
            detail['prediction_response'] = [b-a for a,b in zip(first,second)] if first and second else None
            detail['response_mse'] = (sum((b-a-d)**2 for a,b,d in zip(first,second,delta))/4
                                      if first and second else None)
            detail['prediction_changed'] = any(abs(b-a) > 1e-8 for a,b in zip(first,second)) if first and second else None
            detail['physical_units'] = ['m/s', 'm/s', 'm', 'm']
        grouped[name].append(detail)
    summary = {}
    for name, details in grouped.items():
        if '/coding/' in name:
            summary[name] = {'count': len(details),
                'exact_accuracy': sum(d['correct'] for d in details)/len(details),
                'prediction_unchanged_fraction': sum(d['prediction_unchanged'] for d in details)/len(details),
                'same_reference_count': sum(d['reference_unchanged'] for d in details)}
        else:
            finite = [d['response_mse'] for d in details if d['response_mse'] is not None]
            summary[name] = {'count': len(details), 'finite_response_count': len(finite),
                'response_mse': sum(finite)/len(finite) if len(finite) == len(details) else None,
                'prediction_changed_fraction': sum(d['prediction_changed'] is True for d in details)/len(details),
                'law_test': 'error of predicted state response against independently verified analytical response',
                'change_alone_is_not_law_consistency': True}
    return {'summary': summary, 'records': dict(grouped),
            'candidate_order_invariance': {'status': 'NOT_APPLICABLE', 'reason': 'output generation has no candidates'}}


def _safe_path(root, relative):
    if Path(relative).is_absolute():
        raise ValueError('package paths must be relative')
    path = (root/relative).resolve()
    if root.resolve() not in path.parents:
        raise ValueError('package path escapes root')
    return path


def _checkpoint_path(root, path, input_root='/kaggle/input'):
    path = Path(path)
    path = path.resolve() if path.is_absolute() else (root/path).resolve()
    mount = Path(input_root).resolve()
    if mount not in path.parents or not path.is_file():
        raise PermissionError('checkpoint must be a file on provider input mount')
    try:
        # No truncate/create flags: this probe cannot alter checkpoint bytes.
        descriptor = os.open(path, os.O_WRONLY)
    except OSError as exc:
        if exc.errno not in (errno.EROFS, errno.EACCES, errno.EPERM):
            raise PermissionError('checkpoint read-only status indeterminate') from exc
    else:
        os.close(descriptor)
        raise PermissionError('checkpoint permits write access')
    return path


def _runtime_guard(manifest, package, environ=None, home=None):
    environ = os.environ if environ is None else environ
    if sys.platform != 'linux' or environ.get('KAGGLE_KERNEL_RUN_TYPE') != 'Batch':
        raise PermissionError('independent Linux Kaggle Batch execution required')
    credential_names = ('KAGGLE_KEY','KAGGLE_API_TOKEN','KAGGLE_USERNAME',
        'AWS_SECRET_ACCESS_KEY','GITHUB_TOKEN','GH_TOKEN','HF_TOKEN','GOOGLE_APPLICATION_CREDENTIALS')
    if any(environ.get(key) for key in credential_names):
        raise PermissionError('known credential environment present')
    home = Path.home() if home is None else Path(home)
    for rel in ('.kaggle/kaggle.json','.kaggle/access_token','.kaggle/oauth_credentials.json',
                '.config/kaggle/access_token','.config/gh/hosts.yml','.aws/credentials'):
        if (home/rel).exists():
            raise PermissionError('known credential file present')
    path = _safe_path(package, manifest['runtime_audit_path'])
    if _sha(path) != manifest['runtime_audit_sha256']:
        raise PermissionError('runtime audit pin mismatch')
    audit = _read(path)
    for field in ('all_fits_locked','final_test_selection_frozen','controller_trusted'):
        if audit.get(field) is not True:
            raise PermissionError('evaluation freeze required: '+field)
    trainer_ids = audit.get('trainer_kernel_ids',[])
    evaluator_id = audit.get('evaluator_kernel_id')
    if (audit.get('provider') != 'kaggle' or audit.get('device') != 'cpu' or
            len(trainer_ids) != 2 or len(set(trainer_ids)) != 2 or
            not evaluator_id or evaluator_id in trainer_ids):
        raise PermissionError('two frozen trainers and distinct CPU evaluator required')
    return {'provider_batch_execution':True, 'known_credential_environment_absent':True,
            'known_credential_files_absent':True, 'controller_audit_sha256':_sha(path),
            'protection_scope':'frozen-runtime-isolation','controller_trusted':True,
            'owner_can_change_future_runs':True, 'controller_audit':audit}


def run(package, output):
    # Delay model imports so metric tests require no torch or optimizer modules.
    import torch
    package, output = Path(package), Path(output)
    manifest_path = package/'evaluation-manifest.json'
    manifest = _read(manifest_path)
    if manifest.get('schema') != 'r6-evaluation-v1':
        raise ValueError('unsupported evaluation manifest')
    runtime_audit = _runtime_guard(manifest, package)
    if torch.cuda.is_available():
        raise PermissionError('CPU-only independent execution required')
    runtime_audit['cuda_available'] = False
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('evaluation output must be empty')
    output.mkdir(parents=True, exist_ok=True)
    if not manifest.get('source_manifest'):
        raise ValueError('nonempty source manifest required')
    for rel, expected in manifest['source_manifest'].items():
        if _sha(_safe_path(package, rel)) != expected:
            raise ValueError('source hash mismatch: '+rel)
    from research.r6.architectures.model import make_model, predict
    from research.r6.common import load_checkpoint
    frozen_inputs = {manifest_path:_sha(manifest_path)}
    frozen_inputs.update({_safe_path(package, rel):expected for rel,expected in manifest['source_manifest'].items()})
    rows, dataset_hashes = [], {}
    for item in manifest['datasets']:
        path = _safe_path(package, item['path'])
        if _sha(path) != item['sha256']:
            raise ValueError('dataset hash mismatch')
        current = _read(path)
        current = current['records'] if isinstance(current, dict) else current
        if any(r['split'] != item['split'] or r['split'] not in ('final_iid','final_ood') for r in current):
            raise ValueError('final split identity mismatch')
        rows.extend(current)
        dataset_hashes[item['path']] = item['sha256']
        frozen_inputs[path] = item['sha256']
    if len({r['canonical_key'] for r in rows}) != len(rows):
        raise ValueError('canonical duplicate in protected final data')
    probes = make_probes(rows)
    normalization_path = _safe_path(package, manifest.get('normalization_path', 'normalization.json'))
    if manifest.get('normalization_sha256') and _sha(normalization_path) != manifest['normalization_sha256']:
        raise ValueError('normalization hash mismatch')
    normalization = _read(normalization_path)
    frozen_inputs[normalization_path] = _sha(normalization_path)
    baseline = score(rows, baseline_predictions(rows, normalization))
    references = reference_predictions(rows)
    reference_score = score(rows, references)
    for domain, values in reference_score['domains'].items():
        if domain == 'physics':
            if values['mse'] is None or values['mse'] > 1e-18:
                raise ValueError('independent physical answer verification failed')
        elif values['exact_accuracy'] != 1:
            raise ValueError('independent exact answer verification failed')
    _write(output/'reference-predictions.json', [{'record_id':r['record_id'], 'prediction':p} for r,p in zip(rows,references)])
    config_path = _safe_path(package, manifest.get('config_path', 'config.json'))
    if manifest.get('config_sha256') and _sha(config_path) != manifest['config_sha256']:
        raise ValueError('model config hash mismatch')
    config = _read(config_path)
    frozen_inputs[config_path] = _sha(config_path)
    audit_path = _safe_path(package, manifest['runtime_audit_path'])
    frozen_inputs[audit_path] = manifest['runtime_audit_sha256']
    completed = []
    identities = [(x['fit_id'],x['role']) for x in manifest['checkpoints']]
    if len(set(identities)) != len(identities) or not identities:
        raise ValueError('empty/duplicate checkpoint identities')
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    for checkpoint in manifest['checkpoints']:
        start = time.monotonic()
        path = _checkpoint_path(package, checkpoint['path'])
        if _sha(path) != checkpoint['sha256']:
            raise ValueError('checkpoint hash mismatch')
        model = make_model(checkpoint.get('config', config))
        metadata = load_checkpoint(path, model)
        if _sha(path) != checkpoint['sha256']:
            raise ValueError('checkpoint changed during load')
        model.requires_grad_(False)
        model.to('cpu').eval()
        with torch.inference_mode():
            predictions = predict(model, rows)
            probe_predictions = predict(model, [x['row'] for x in probes])
        if any(p.requires_grad or p.grad is not None for p in model.parameters()):
            raise PermissionError('model gradient state is not disabled')
        if _sha(path) != checkpoint['sha256']:
            raise ValueError('checkpoint changed during evaluation')
        frozen_inputs[path] = checkpoint['sha256']
        metrics = score(rows, predictions)
        robustness = score_probes(rows, predictions, probes, probe_predictions)
        slug = checkpoint['fit_id']+'-'+checkpoint['role']
        if not slug.replace('-','').replace('_','').isalnum():
            raise ValueError('unsafe checkpoint identity')
        raw = [{'record_id':r['record_id'], 'domain':r['domain'], 'split':r['split'],
                'topic':r['topic'], 'target':r['target'], 'prediction':p,
                'checkpoint_sha256':checkpoint['sha256']} for r,p in zip(rows,predictions)]
        failures = [r for r in raw if (r['prediction'].get('answer') != str(r['target']['answer'])
                    if r['domain'] != 'physics' else _state(r['prediction']) != r['target']['state'])]
        _write(output/(slug+'-predictions.json'), raw)
        _write(output/(slug+'-perturbations.json'), robustness)
        _write(output/(slug+'-failures.json'), failures)
        completed.append({'fit_id':checkpoint['fit_id'], 'role':checkpoint['role'],
            'checkpoint_sha256':checkpoint['sha256'], 'metrics':metrics,
            'perturbation_summary':robustness['summary'],
            'parameter_count':sum(p.numel() for p in model.parameters()),
            'seconds':time.monotonic()-start,
            'prediction_file':slug+'-predictions.json', 'failure_file':slug+'-failures.json',
            'checkpoint_metadata':metadata})
    for path, expected in frozen_inputs.items():
        if _sha(path) != expected:
            raise ValueError('frozen source/data/checkpoint changed during evaluation: '+str(path))
    runtime_audit['source_dataset_checkpoint_hashes_rechecked_after_evaluation'] = True
    runtime_audit['checkpoint_write_open_denied_paths'] = [str(_checkpoint_path(package,x['path'])) for x in manifest['checkpoints']]
    runtime_audit['model_requires_grad'] = False
    receipt = {'schema':'r6-evaluation-receipt-v1', 'status':'COMPLETED', 'device':'cpu',
        'optimizer_updates_executed':0, 'gradients_enabled':torch.is_grad_enabled(),
        'evaluation_manifest_sha256':_sha(manifest_path), 'dataset_sha256':dataset_hashes,
        'runtime_audit':runtime_audit,
        'count':len(rows), 'evaluations':completed, 'training_prior_baselines':baseline,
        'external_exact_reference_baselines':reference_score,
        'historical_r57c_on_r6':{'status':'UNSUPPORTED_INPUT_AND_OUTPUT_REPRESENTATION',
             'reason':'Historical candidate-index/digit/velocity-delta heads do not predict R6 integer strings and four-state motion; no synthetic zero score assigned'},
        'candidate_order_invariance':{'status':'NOT_APPLICABLE','reason':'No answer candidates'},
        'isolation_claim':'Runtime/mount/control-plane evidence must be supplied by the independent cloud controller',
        'final_data_policy':'one frozen evaluation execution after all supplied fits/corrections locked'}
    receipt['artifacts'] = {p.name:_sha(p) for p in sorted(output.glob('*.json'))}
    _write(output/'evaluation_receipt.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.package, args.output)


if __name__ == '__main__':
    main()
