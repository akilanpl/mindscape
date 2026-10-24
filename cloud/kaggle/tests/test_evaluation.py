"""Development-only tests: no fitting, optimizer or cloud-execution simulation."""
from copy import deepcopy
import json

import pytest

from cloud.kaggle.evaluation.worker import (
    baseline_results, evaluate, overlap_check, perturbations, require_isolation,
    run, verified_path,
)
from mindscape.r57.evaluation import analytic


def rows():
    problems = [
        {'domain': 'math', 'operation': 'add_digit', 'left': 9, 'right': 2, 'incoming': 0},
        {'domain': 'coding', 'operation': 'rename', 'source': 'x = 1', 'old_name': 'x',
         'candidates': ['aa', 'zz', 'mm'], 'requested_name': 'zz'},
        {'domain': 'physics', 'operation': 'force', 'mass': 2., 'dt': .5,
         'force': [4., -8.], 'velocity': [0., 0.]},
    ]
    return [{'problem': p, 'labels': analytic(p), 'family': p['domain']} for p in problems]


def test_metrics_raw_predictions_are_preserved_without_repair():
    examples = rows()
    def wrong(problem):
        if problem['domain'] == 'coding':
            return {'coding_index': 0}
        return analytic(problem)
    result = evaluate(wrong, examples)
    assert result['metrics']['math']['joint_accuracy'] == 1
    assert result['metrics']['physics']['mse'] == 0
    assert result['metrics']['coding']['accuracy'] == 0
    record = result['records'][1]
    assert record['prediction'] == record['raw_output'] == {'coding_index': 0}
    assert record['reference'] == {'coding_index': 1}
    assert record['repaired'] is False


def test_normalized_leakage_and_duplicates_rejected():
    train = [rows()[0]]
    same = deepcopy(train)
    same[0]['problem']['carry_in'] = same[0]['problem'].pop('incoming')
    same[0]['problem']['concept_ids'] = [99]
    with pytest.raises(ValueError, match='overlap'):
        overlap_check(train, same)
    with pytest.raises(ValueError, match='Duplicate normalized evaluation'):
        overlap_check([], rows() + [rows()[0]])


def test_sensitivity_and_candidate_permutation_controls():
    examples = rows(); before = deepcopy(examples)
    probes = perturbations(analytic, examples)
    assert examples == before
    indexed = {r['probe']: r for r in probes}
    assert indexed['mass_times_two']['prediction']['physics_delta'] == [.5, -1.]
    assert indexed['dt_times_two']['prediction']['physics_delta'] == [2., -4.]
    assert indexed['incoming_flip']['prediction'] == {'math_digit': 2, 'math_carry': 1}
    assert indexed['candidate_rotation_1']['same_selected_name'] is True
    assert indexed['requested_name_change']['same_selected_name'] is False
    assert all(r['prediction'] == r['analytic_reference'] for r in probes)


def test_shortcut_does_not_pass_candidate_order_check():
    probes = perturbations(lambda p: {'coding_index': 0}, [rows()[1]])
    assert not any(r['same_selected_name'] for r in probes if r['probe'].startswith('candidate_rotation'))


def test_baselines_are_separate_and_training_only():
    training = rows(); evaluation = deepcopy(training)
    evaluation[0]['problem']['left'] = 0
    evaluation[0]['labels'] = analytic(evaluation[0]['problem'])
    result = baseline_results(training, evaluation)
    assert result['analytic_rule']['metrics']['math']['joint_accuracy'] == 1
    assert result['training_majority_constant']['metrics']['math']['joint_accuracy'] == 0
    assert 'non-neural' in result['classification']


def test_hash_tampering_fails_before_decode(tmp_path):
    path = tmp_path / 'input.json'; path.write_text('invalid JSON')
    with pytest.raises(ValueError, match='hash mismatch'):
        verified_path(tmp_path, {'path': 'input.json', 'sha256': '0' * 64})


def test_json_flags_cannot_authenticate_access_isolation():
    evidence = {key: True for key in (
        'control_plane_verified', 'separate_execution', 'evaluation_data_readonly',
        'checkpoint_inputs_readonly', 'trainer_denied_evaluation_data',
        'no_training_credentials', 'internet_disabled', 'attestation_authenticated',
    )}
    evidence.update(trainer_principal='trainer', evaluator_principal='evaluator',
                    controller_principal='controller', execution_id='123', provider_evidence_uri='evidence')
    with pytest.raises(PermissionError, match='no authenticated provider ACL verifier'):
        require_isolation(evidence)


def test_entrypoint_fails_closed_before_unpinned_data_or_output(tmp_path):
    import hashlib
    evidence = tmp_path / 'isolation.json'; evidence.write_text('{}')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'schema': 'mindscape-independent-evaluation-v1',
        'isolation_evidence': {'path': evidence.name,
            'sha256': hashlib.sha256(evidence.read_bytes()).hexdigest()}}))
    output = tmp_path / 'output'
    with pytest.raises(PermissionError, match='INDEPENDENT_ENVIRONMENT_UNVERIFIED'):
        run(manifest, output)
    assert not output.exists()


def test_frozen_training_repeats_are_reported_without_changing_protocol():
    training = [rows()[0], rows()[0]]
    audit = overlap_check(training, [rows()[1]])
    assert audit['training_count'] == 2
    assert audit['training_normalized_duplicates'] == 1
    assert len(training) == 2


def controller_evidence():
    def role(kernel, cpu=False):
        return {'kernel_id': kernel, 'version': 1, 'source_sha256': 'a' * 64,
                'remote_metadata_sha256': 'b' * 64, 'control_plane_source_verified': True,
                'credential_access_code_absent': True,
                'metadata': {'id': kernel, 'is_private': True, 'enable_internet': False,
                             'enable_gpu': not cpu, 'enable_tpu': False}}
    trainer = role('owner/trainer'); trainer.update(payload_evaluation_records=0, evaluation_inputs_attached=False)
    evaluator = role('owner/evaluator', True)
    evaluator.update(source_identity_scope='evaluator-core-file', source_path='cloud/kaggle/evaluation/worker.py',
                     core_source_files={'cloud/kaggle/evaluation/worker.py': 'a' * 64})
    return {'provider': 'kaggle', 'protection_scope': 'frozen-runtime-isolation',
            'trainer': trainer, 'evaluator': evaluator,
            'checkpoint_export_hash_verified': True, 'checkpoint_input_version': 'owner/trainer/1',
            'controller_trusted': True}


def test_same_owner_distinct_executions_are_permitted_under_trusted_controller():
    from cloud.kaggle.evaluation.isolation import validate_controller_evidence
    assert validate_controller_evidence(controller_evidence()) is True


def test_trainer_eval_attachment_and_remote_source_unverified_fail():
    from cloud.kaggle.evaluation.isolation import validate_controller_evidence
    evidence = controller_evidence(); evidence['trainer']['evaluation_inputs_attached'] = True
    with pytest.raises(PermissionError, match='expose evaluation'):
        validate_controller_evidence(evidence)
    evidence = controller_evidence(); evidence['trainer']['control_plane_source_verified'] = False
    with pytest.raises(PermissionError, match='CLI-pulled'):
        validate_controller_evidence(evidence)


def test_readonly_guard_checks_effective_access_without_changing_bytes(tmp_path):
    from cloud.kaggle.evaluation.isolation import readonly_input
    checkpoint = tmp_path / 'checkpoint.json'; checkpoint.write_bytes(b'original')
    with pytest.raises(PermissionError, match='permits writes'):
        readonly_input(checkpoint, tmp_path)
    assert checkpoint.read_bytes() == b'original'


def test_controller_evidence_digest_cannot_be_changed(tmp_path):
    from cloud.kaggle.evaluation.isolation import evidence_digest, kaggle_verifier
    evidence = controller_evidence()
    verifier = kaggle_verifier(evidence_digest(evidence), [tmp_path / 'checkpoint'],
                              environ={'KAGGLE_KERNEL_RUN_TYPE': 'Batch'}, home=tmp_path)
    evidence['trainer']['version'] = 2
    with pytest.raises(PermissionError, match='digest mismatch'):
        verifier(evidence)


def test_known_credentials_fail_without_printing_values(tmp_path):
    from cloud.kaggle.evaluation.isolation import credential_guard
    with pytest.raises(PermissionError, match='credential environment') as exc:
        credential_guard({'GH_TOKEN': 'do-not-print'}, tmp_path)
    assert 'do-not-print' not in str(exc.value)


def test_verifier_binds_actual_core_bytes_and_readonly_check(tmp_path, monkeypatch):
    import errno
    import hashlib
    from cloud.kaggle.evaluation import isolation
    core = tmp_path / 'core.py'; core.write_text('CORE = True\n')
    evidence = controller_evidence()
    sha = hashlib.sha256(core.read_bytes()).hexdigest()
    evidence['evaluator'].update(source_path='core.py', source_sha256=sha,
                                 core_source_files={'core.py': sha})
    checkpoint = tmp_path / 'checkpoint.json'; checkpoint.write_text('{}')
    def readonly_open(*args, **kwargs):
        raise OSError(errno.EROFS, 'read-only filesystem')
    monkeypatch.setattr(isolation.os, 'open', readonly_open)
    verifier = isolation.kaggle_verifier(isolation.evidence_digest(evidence), [checkpoint],
        input_root=tmp_path, core_root=tmp_path,
        environ={'KAGGLE_KERNEL_RUN_TYPE': 'Batch'}, home=tmp_path)
    assert verifier(evidence) is True
    core.write_text('CORE = False\n')
    with pytest.raises(PermissionError, match='core source hash mismatch'):
        verifier(evidence)


def test_receipt_preserves_exact_distribution_hashes_without_neural_inference(tmp_path, monkeypatch):
    import hashlib
    import sys
    from types import SimpleNamespace
    from pathlib import Path
    from cloud.kaggle.evaluation import worker
    def pin(path):
        return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    training = tmp_path / 'training.json'; training.write_text(json.dumps([rows()[0]]))
    final = tmp_path / 'final.json'; final.write_text(json.dumps(rows()[1:]))
    checkpoint = tmp_path / 'checkpoint.json'; checkpoint.write_text('{}')
    isolation = tmp_path / 'isolation.json'; isolation.write_text('{}')
    source_files = [pin(Path(module.__file__)) for name, module in tuple(sys.modules.items())
                    if name.startswith('mindscape.') and getattr(module, '__file__', None)]
    entry = {**pin(checkpoint), 'seed': 17, 'condition': 'graph-concept-on', 'version': 'R57c'}
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'schema': 'mindscape-independent-evaluation-v1',
        'source_files': source_files, 'data_files': [pin(training), pin(final)],
        'checkpoints': [entry], 'isolation_evidence': pin(isolation),
        'evaluations': [{'seed': 17, 'training_path': str(training), 'evaluation_path': str(final),
                         'distribution_id': 'test-fixture-only', 'historical_evaluation_reuse': True}]}))
    # Exercises receipt assembly and hash provenance, not cloud execution or a model.
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(__version__='test-fixture-only',
        set_num_threads=lambda n: None, use_deterministic_algorithms=lambda value: None))
    monkeypatch.setattr(worker, 'require_isolation', lambda *args, **kwargs: None)
    monkeypatch.setattr(worker, 'load_model', lambda *args: analytic)
    receipt = worker.run(manifest, tmp_path / 'results')
    block = receipt['results'][0]
    assert block['evaluation_sha256'] == pin(final)['sha256']
    assert block['training_sha256'] == pin(training)['sha256']
    assert block['source_files'] == source_files
    assert block['historical_evaluation_reuse'] is True
