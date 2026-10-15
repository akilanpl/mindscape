"""Read-only runner fail-closed tests; no optimizer construction or updates."""
import hashlib
import json
from pathlib import Path
import pytest
from mindscape.r57 import runner


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def test_mps_refusal_precedes_directory_or_model(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'verify_identity',lambda out:{})
    def refuse():
        raise RuntimeError('NATIVE_MPS_EXECUTION_REQUIRED')
    monkeypatch.setattr(runner,'require_mps',refuse)
    with pytest.raises(RuntimeError,match='NATIVE_MPS_EXECUTION_REQUIRED'):
        runner.fit(tmp_path,17,'graph-concept-on')
    assert list(tmp_path.iterdir())==[]


def test_source_hash_mutation_rejected(tmp_path):
    root=Path(runner.__file__).resolve().parents[3]
    manifest=tmp_path/'data-v1/manifest.json';write(manifest,{'identity':'test'})
    frozen={'status':'CANDIDATE_FROZEN','source_root':str(root),
            'source_hashes':{'src/mindscape/r57/runner.py':'0'*64},
            'data_manifest_sha256':runner.sha(manifest)}
    write(tmp_path/'candidate-freeze.json',frozen)
    with pytest.raises(RuntimeError,match='Source changed'):
        runner.verify_identity(tmp_path)


def test_resume_requires_real_checkpoint_hash_and_steps(tmp_path,monkeypatch):
    run=tmp_path/'checkpoints/17-graph-concept-on';run.mkdir(parents=True)
    write(tmp_path/'candidate-freeze.json',{'source_commit':'fixed'})
    write(tmp_path/'data-v1/train/17.json',{'data':'fixed'})
    monkeypatch.setattr(runner,'verify_identity',lambda out:{'source_commit':'fixed'})
    (run/'initial.json').write_text('initial');(run/'final.json').write_text('final')
    log=tmp_path/'losses/17-graph-concept-on.jsonl';log.parent.mkdir()
    log.write_text(''.join(json.dumps({'update':i})+'\n' for i in range(1,2001)))
    result={'status':'COMPLETED_EXPLORATORY','optimizer_updates':2000,'optimizer_steps':2000,'step_post_hook_count':2000,
            'seed':17,'condition':'graph-concept-on','source_commit':'fixed',
            'candidate_freeze_sha256':runner.sha(tmp_path/'candidate-freeze.json'),
            'training_sha256':runner.sha(tmp_path/'data-v1/train/17.json'),
            'loss_sha256':runner.sha(log), 'initial_sha256':runner.sha(run/'initial.json'),
            'checkpoint_sha256':runner.sha(run/'final.json')}
    write(run/'result.json',result)
    assert runner.completed(run)['optimizer_updates']==2000
    result['optimizer_updates']=1999;write(run/'result.json',result)
    with pytest.raises(RuntimeError,match='Incomplete'):runner.completed(run)
    result['optimizer_updates']=2000;write(run/'result.json',result)
    log.write_text(log.read_text()+'{}\n')
    with pytest.raises(RuntimeError,match='Loss evidence changed'):runner.completed(run)
    result['loss_sha256']=runner.sha(log);write(run/'result.json',result)
    with pytest.raises(RuntimeError,match='Loss update sequence'):runner.completed(run)
    log.write_text(''.join(json.dumps({'update':i})+'\n' for i in range(1,2001)))
    result['loss_sha256']=runner.sha(log);write(run/'result.json',result)
    (run/'final.json').write_text('mutated')
    with pytest.raises(RuntimeError,match='checkpoint changed'):runner.completed(run)


def test_fallback_opt_in_rejected_before_torch(tmp_path,monkeypatch):
    import builtins
    monkeypatch.setenv('PYTORCH_ENABLE_MPS_FALLBACK','1')
    original=builtins.__import__
    def no_torch(name,*a,**k):
        if name=='torch':raise AssertionError('Torch import before fallback rejection')
        return original(name,*a,**k)
    monkeypatch.setattr(builtins,'__import__',no_torch)
    with pytest.raises(RuntimeError,match='CPU fallback forbidden'):
        runner.require_mps()
