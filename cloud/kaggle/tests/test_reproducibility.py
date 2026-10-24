import json
import pytest
from cloud.kaggle.reproducibility import snapshot, sha, verify_export


def test_snapshot_hashes_bytes_excludes_git_and_cache(tmp_path):
    (tmp_path/'a').write_bytes(b'private research')
    (tmp_path/'.git').mkdir()
    (tmp_path/'.git'/'config').write_text('excluded')
    (tmp_path/'__pycache__').mkdir()
    (tmp_path/'__pycache__'/'x').write_bytes(b'excluded')
    assert snapshot(tmp_path) == {'a':sha(tmp_path/'a')}


def test_export_rejects_partial_budget_before_artifact_use(tmp_path):
    (tmp_path/'result.json').write_text(json.dumps({'optimizer_updates':1999}))
    with pytest.raises(ValueError,match='budget'):
        verify_export(tmp_path)


def test_export_rejects_false_exposure_counts(tmp_path):
    (tmp_path/'result.json').write_text(json.dumps(dict(optimizer_updates=2000,optimizer_steps=2000,
        status='COMPLETED_EXPLORATORY',device='cuda:0',seed=17,condition='matched-gru',
        source_commit='d54ff8fe64f30c53b9662ddba90691338aad05f6',step_post_hook_count=2000,adam_steps_by_parameter={'x':2},
        gradient_exposures_by_parameter={'x':3})))
    with pytest.raises(ValueError,match='exposure'):
        verify_export(tmp_path)
