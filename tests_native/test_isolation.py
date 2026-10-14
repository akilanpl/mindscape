import hashlib
import json
from pathlib import Path
import pytest
from mindscape.r56.isolation import checked_file,stage_training,verify_stage,container_command,run_probe

IMAGE='python@sha256:'+'a'*64


def make_package(tmp_path):
    package=tmp_path/'package';package.mkdir()
    raw=json.dumps([{'split':'train','problem':{'left':i}} for i in range(512)]).encode()
    (package/'train.json').write_bytes(raw)
    (package/'targets.json').write_text('PRIVATE_EVALUATOR_CANARY')
    from mindscape.r56.curriculum import digest
    manifest={'index':{'17':{'training_file':'train.json','sha256':hashlib.sha256(raw).hexdigest()}}}
    manifest['package_sha256']=digest(manifest)
    (package/'manifest.json').write_text(json.dumps(manifest))
    return package


def test_stage_does_not_mount_or_copy_heldout_answers(tmp_path):
    package=make_package(tmp_path);stage=tmp_path/'stage'
    stage_training(package,stage,17,json.loads((package/'manifest.json').read_text())['package_sha256'])
    assert set(p.name for p in stage.iterdir())=={'train.json','staging.json'}
    assert 'PRIVATE_EVALUATOR_CANARY' not in ''.join(p.read_text() for p in stage.iterdir())
    verify_stage(stage)
    (stage/'train.json').chmod(0o644);(stage/'train.json').write_text('tampered')
    with pytest.raises(ValueError,match='tampered'):verify_stage(stage)


@pytest.mark.parametrize('name',['../targets.json','/etc/passwd','a/../../targets.json'])
def test_path_traversal_rejected(tmp_path,name):
    with pytest.raises(ValueError):checked_file(tmp_path,name)


def test_symlink_and_directory_symlink_rejected(tmp_path):
    target=tmp_path/'targets';target.write_text('secret')
    (tmp_path/'link').symlink_to(target)
    with pytest.raises(ValueError):checked_file(tmp_path,'link')
    (tmp_path/'dir').mkdir();(tmp_path/'dir'/'data').write_text('data')
    (tmp_path/'linked-dir').symlink_to(tmp_path/'dir')
    with pytest.raises(ValueError):checked_file(tmp_path,'linked-dir/data')


def test_container_contract_seals_mounts_and_rejects_evaluator_mount(tmp_path):
    package=make_package(tmp_path);stage=tmp_path/'stage';stage_training(package,stage,17,json.loads((package/'manifest.json').read_text())['package_sha256'])
    probe=tmp_path/'probe.py';probe.write_text('print(1)')
    args=container_command(IMAGE,stage,probe)
    assert '--network' in args and args[args.index('--network')+1]=='none'
    assert '--read-only' in args and '10001:10001' in args and '--cap-drop' in args
    assert not any('targets.json' in arg for arg in args)
    with pytest.raises(ValueError):container_command('python:latest',stage,probe)
    with pytest.raises(ValueError):container_command(IMAGE,stage,probe,evaluator_data=package)


def test_no_runtime_is_explicitly_blocked(tmp_path,monkeypatch):
    import mindscape.r56.isolation as module
    package=make_package(tmp_path);stage=tmp_path/'stage';stage_training(package,stage,17,json.loads((package/'manifest.json').read_text())['package_sha256'])
    probe=tmp_path/'probe.py';probe.write_text('print(1)')
    monkeypatch.setattr(module.shutil,'which',lambda name:None)
    result=run_probe(IMAGE,stage,probe)
    assert result['status']=='BLOCKED' and result['qualified'] is False


def test_launch_closes_fds_and_uses_clean_environment(tmp_path,monkeypatch):
    import mindscape.r56.isolation as module
    package=make_package(tmp_path);stage=tmp_path/'stage';stage_training(package,stage,17,json.loads((package/'manifest.json').read_text())['package_sha256'])
    probe=tmp_path/'probe.py';probe.write_text('print(1)')
    monkeypatch.setenv('EVAL_SECRET','must-not-be-inherited')
    monkeypatch.setattr(module.shutil,'which',lambda name:'/trusted/docker')
    def launch(command,**kwargs):
        assert kwargs['close_fds'] is True and 'EVAL_SECRET' not in kwargs['env']
        assert 'shell' not in kwargs and command[0]=='docker'
        class Result:returncode=1;stdout='';stderr='contract only; no container executed'
        return Result()
    monkeypatch.setattr(module.subprocess,'run',launch)
    assert run_probe(IMAGE,stage,probe)['qualified'] is False


def test_evaluator_has_distinct_readonly_data_checkpoint_mounts(tmp_path):
    package=make_package(tmp_path);stage=tmp_path/'stage';stage_training(package,stage,17,json.loads((package/'manifest.json').read_text())['package_sha256'])
    probe=tmp_path/'probe.py';probe.write_text('print(1)')
    vault=tmp_path/'vault';vault.mkdir();checkpoints=tmp_path/'checkpoints';checkpoints.mkdir()
    args=container_command(IMAGE,stage,probe,role='evaluator',evaluator_data=vault,checkpoint_dir=checkpoints)
    assert any('dst=/sealed-evaluation,readonly' in arg for arg in args)
    assert any('dst=/checkpoints,readonly' in arg for arg in args)
    (vault/'escape').symlink_to(package/'targets.json')
    with pytest.raises(ValueError):container_command(IMAGE,stage,probe,role='evaluator',evaluator_data=vault,checkpoint_dir=checkpoints)


def test_nonroot_stage_is_readable_and_probe_symlink_rejected(tmp_path):
    package=make_package(tmp_path);stage=tmp_path/'stage';stage_training(package,stage,17,json.loads((package/'manifest.json').read_text())['package_sha256'])
    assert stage.stat().st_mode & 0o555 == 0o555
    target=tmp_path/'probe.py';target.write_text('print(1)');link=tmp_path/'probe-link.py';link.symlink_to(target)
    with pytest.raises(ValueError):container_command(IMAGE,stage,link)


def test_probe_identity_and_positive_control_are_required(tmp_path,monkeypatch):
    import mindscape.r56.isolation as module
    package=make_package(tmp_path);stage=tmp_path/'stage';stage_training(package,stage,17,json.loads((package/'manifest.json').read_text())['package_sha256'])
    probe=tmp_path/'probe.py';probe.write_text('print(1)')
    with pytest.raises(ValueError,match='identity'):
        module.run_isolation_pair(IMAGE,stage,probe,package,package,'a'*64,'b'*64,'0'*64)
    monkeypatch.setattr(module,'run_probe',lambda *a:{'boundary_probe_passed':True,'qualified':False,'status':'TRAINER_PROBE_ONLY'})
    def negative(*a,**k):
        class Result:returncode=0;stdout='{"target_readable": false, "checkpoint_readable": false}'
        return Result()
    monkeypatch.setattr(module.subprocess,'run',negative)
    receipt=module.run_isolation_pair(IMAGE,stage,probe,package,package,'a'*64,'b'*64,module.sha(probe))
    assert receipt['boundary_probe_passed'] is False and receipt['qualified'] is False


def test_manifest_rehash_cannot_override_external_proposal_pin(tmp_path):
    from mindscape.r56.curriculum import digest
    package=make_package(tmp_path)
    manifest=json.loads((package/'manifest.json').read_text());pinned=manifest.pop('package_sha256')
    manifest['extra']='tampering';manifest['package_sha256']=digest(manifest)
    (package/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='identity'):
        stage_training(package,tmp_path/'stage',17,pinned)


def test_probe_missing_symlink_exercise_is_failure_not_success():
    source=(Path(__file__).resolve().parents[1]/'deploy/r56-isolation/probe.py').read_text()
    assert "except OSError:result['symlink_denied']=False" in source
    assert "'checkpoint_denied':denied('/checkpoints/checkpoint.json')" in source
