import math
import errno
import json
import os

import pytest

from research.r6.datasets import coding, mathematics, physics
from research.r6.evaluation.worker import (baseline_predictions, make_probes,
    reference_predictions, score, score_probes, _checkpoint_path, _safe_path,
    _runtime_guard, _sha)


def cases():
    return (mathematics.generate('final_iid', 71, 6) + coding.generate('final_iid', 71, 6) +
            physics.generate('final_iid', 71, 6) + mathematics.generate('final_ood', 71, 6) +
            coding.generate('final_ood', 71, 6) + physics.generate('final_ood', 71, 6))


def test_reference_engines_and_stratification():
    rows = cases()
    metrics = score(rows, reference_predictions(rows))
    assert metrics['count'] == 36
    assert metrics['domains']['coding']['exact_accuracy'] == 1
    assert metrics['domains']['mathematics']['exact_accuracy'] == 1
    assert metrics['domains']['physics']['mse'] == pytest.approx(0, abs=1e-20)
    assert len(metrics['per_split']) == 6
    assert metrics['math_carry_borrow']


def test_exact_answer_accuracy_does_not_coerce_bad_strings():
    rows = coding.generate('final_iid', 71, 3)
    predictions = [{'answer': rows[0]['target']['answer']},
                   {'answer': ' '+rows[1]['target']['answer']}, {'answer': None}]
    result = score(rows, predictions)['domains']['coding']
    assert result['correct'] == 1
    assert result['exact_accuracy'] == pytest.approx(1/3)
    assert result['invalid_prediction_count'] == 1


def test_physics_error_per_axis_and_invalids_are_not_omitted():
    rows = physics.generate('final_iid', 71, 2)
    predictions = [{'state': [v+(i+1) for i,v in enumerate(row['target']['state'])]} for row in rows]
    result = score(rows,predictions)['domains']['physics']
    assert result['mse'] == 7.5
    assert result['mae'] == 2.5
    assert result['mse_per_dimension'] == pytest.approx([1,4,9,16])
    predictions[0]['state'][0] = math.nan
    result = score(rows,predictions)['domains']['physics']
    assert result['mse'] is None and result['invalid_prediction_count'] == 1
    assert result['finite_only_diagnostics']['mse'] == 7.5


def test_missing_predictions_and_duplicate_records_rejected():
    rows = coding.generate('final_iid', 71, 2)
    with pytest.raises(ValueError, match='count mismatch'):
        score(rows, [{'answer':'0'}])
    with pytest.raises(ValueError, match='duplicate'):
        score([rows[0],rows[0]], [{'answer':'0'}]*2)


def test_baselines_require_train_frozen_priors():
    rows = cases()
    norm = {'training_majority_answers':{'mathematics':'42','coding':'7'},
            'physics_mean_state':[1,2,3,4]}
    preds = baseline_predictions(rows,norm)
    assert preds[0] == {'answer':'42'}
    assert preds[6] == {'answer':'7'}
    assert preds[12] == {'state':[1,2,3,4]}
    with pytest.raises(ValueError, match='missing'):
        baseline_predictions(rows,{})


def test_reference_interventions_have_zero_response_error():
    rows = cases()
    probes = make_probes(rows)
    result = score_probes(rows, reference_predictions(rows), probes,
                          reference_predictions([p['row'] for p in probes]))
    assert result['candidate_order_invariance']['status'] == 'NOT_APPLICABLE'
    for name, value in result['summary'].items():
        if '/physics/' in name:
            assert value['response_mse'] == pytest.approx(0, abs=1e-20)
            assert value['change_alone_is_not_law_consistency']
        else:
            assert value['exact_accuracy'] == 1


def test_changing_wrong_physics_prediction_does_not_pass_law():
    rows = physics.generate('final_iid',71,2)
    probes = make_probes(rows)
    original = [{'state':[0,0,0,0]} for _ in rows]
    altered = [{'state':[.123,.456,.789,.321]} for _ in probes]
    result = score_probes(rows,original,probes,altered)
    for value in result['summary'].values():
        assert value['prediction_changed_fraction'] == 1
        assert value['response_mse'] > 0


def test_renaming_invariance_does_not_hide_wrong_answers():
    rows = coding.generate('final_iid',71,4)
    probes = make_probes(rows)
    result = score_probes(rows,[{'answer':'WRONG'}]*len(rows), probes,
                          [{'answer':'WRONG'}]*len(probes))
    for name,value in result['summary'].items():
        assert value['exact_accuracy'] == 0
        assert value['prediction_unchanged_fraction'] == 1


def test_package_paths_cannot_escape_or_be_absolute(tmp_path):
    with pytest.raises(ValueError,match='relative'):
        _safe_path(tmp_path,str(tmp_path/'config.json'))
    with pytest.raises(ValueError,match='escapes'):
        _safe_path(tmp_path,'../config.json')


def test_checkpoint_requires_actual_write_denial_on_mount(tmp_path,monkeypatch):
    mount=tmp_path/'input';mount.mkdir()
    checkpoint=mount/'checkpoint.json';checkpoint.write_text('{}')
    with pytest.raises(PermissionError,match='permits write'):
        _checkpoint_path(tmp_path,checkpoint,input_root=mount)
    seen=[]
    def denied(path, flags):
        seen.append(flags)
        raise OSError(errno.EROFS,'read-only')
    monkeypatch.setattr(os,'open',denied)
    assert _checkpoint_path(tmp_path,checkpoint,input_root=mount)==checkpoint
    assert seen==[os.O_WRONLY]
    assert checkpoint.read_text()=='{}'
    with pytest.raises(PermissionError,match='provider input'):
        _checkpoint_path(tmp_path,tmp_path/'elsewhere.json',input_root=mount)


def test_runtime_guard_requires_pinned_freeze_and_no_credentials(tmp_path,monkeypatch):
    from research.r6.evaluation import worker
    monkeypatch.setattr(worker.sys,'platform','linux')
    path=tmp_path/'runtime-audit.json'
    audit={'provider':'kaggle','device':'cpu','all_fits_locked':True,
           'final_test_selection_frozen':True,'controller_trusted':True,
           'trainer_kernel_ids':['owner/train1','owner/train2'],
           'evaluator_kernel_id':'owner/eval'}
    path.write_text(json.dumps(audit))
    manifest={'runtime_audit_path':path.name,'runtime_audit_sha256':_sha(path)}
    env={'KAGGLE_KERNEL_RUN_TYPE':'Batch'}
    result=_runtime_guard(manifest,tmp_path,environ=env,home=tmp_path)
    assert result['known_credential_environment_absent']
    assert result['owner_can_change_future_runs']
    with pytest.raises(PermissionError,match='credential environment'):
        _runtime_guard(manifest,tmp_path,environ={**env,'KAGGLE_API_TOKEN':'test-secret'},home=tmp_path)
    path.write_text(json.dumps({**audit,'final_test_selection_frozen':False}))
    manifest['runtime_audit_sha256']=_sha(path)
    with pytest.raises(PermissionError,match='freeze'):
        _runtime_guard(manifest,tmp_path,environ=env,home=tmp_path)


def test_runtime_guard_rejects_digest_mismatch(tmp_path,monkeypatch):
    from research.r6.evaluation import worker
    monkeypatch.setattr(worker.sys,'platform','linux')
    path=tmp_path/'runtime-audit.json';path.write_text('{}')
    with pytest.raises(PermissionError,match='pin mismatch'):
        _runtime_guard({'runtime_audit_path':path.name,'runtime_audit_sha256':'0'*64},
                       tmp_path,environ={'KAGGLE_KERNEL_RUN_TYPE':'Batch'},home=tmp_path)
