import ast
import base64
import copy
import hashlib
import io
import json
from pathlib import Path
import zipfile

import pytest

from research.r6.cloud.evaluate_campaign import prepare, _metadata_matches


def fixtures(tmp_path):
    repo=Path(__file__).resolve().parents[3]
    data=tmp_path/'data';(data/'protected').mkdir(parents=True)
    for split in ('final_iid','final_ood'):
        (data/'protected'/f'{split}.json').write_text(json.dumps([{'split':split}]))
    (data/'normalization.json').write_text(json.dumps({'computed_from':'train-only'}))
    checkpoints=[]
    for fit in ('raw','scaled'):
        for role in ('initial','best'):
            path=tmp_path/f'{fit}-{role}.json';path.write_text(json.dumps({'fit':fit,'role':role}))
            checkpoints.append({'fit_id':fit,'role':role,'path':str(path),
                'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'config':{'width':48},
                'trainer_kernel_id':'owner/train-'+fit})
    dataset={'dataset_id':'owner/checkpoints','version':1,'is_private':True,
        'download_verified':True,'provider_metadata_sha256':'a'*64,
        'files_sha256':{x['fit_id']+'-'+x['role']+'.json':x['sha256'] for x in checkpoints}}
    trainers={'selected_configs_frozen':True,'budget_actual_updates':1200,'trainers':[]}
    for fit in ('raw','scaled'):
        kernel='owner/train-'+fit
        trainers['trainers'].append({'kernel_id':kernel,'source_sha256':'b'*64,
            'metadata':{'id':kernel,'is_private':True,'enable_internet':False,'enable_gpu':True,
                'kernel_sources':[],'dataset_sources':[]},'source_verified':True,
            'payload_files_sha256':{'train.json':'d'*64},'evaluation_data_absent':True,
            'fit_receipt_sha256':'e'*64})
    return {'repo':repo,'dataset_root':data,'checkpoint_index':checkpoints,
        'dataset_receipt':dataset,'trainer_evidence':trainers,'owner':'owner','slug':'r6-eval',
        'output':tmp_path/'package'}


def unpack_launcher(path):
    tree=ast.parse(path.read_text())
    assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and
        isinstance(n.targets[0],ast.Name) and n.targets[0].id=='payload')
    return zipfile.ZipFile(io.BytesIO(base64.b64decode(ast.literal_eval(assignment.value.args[0]))))


def test_prepared_private_cpu_payload_and_pins(tmp_path):
    inputs=fixtures(tmp_path)
    result=prepare(**inputs)
    metadata=json.loads((inputs['output']/'kernel-metadata.json').read_text())
    assert metadata['dataset_sources']==['owner/checkpoints/1']
    assert metadata['enable_gpu'] is False and metadata['enable_internet'] is False
    assert metadata['is_private'] is True and metadata['kernel_sources']==[]
    with unpack_launcher(inputs['output']/'kernel.py') as archive:
        names=archive.namelist()
        assert 'research/r6/evaluation/worker.py' in names
        assert 'protected/final_iid.json' in names
        assert not any('/training/' in n or n=='train.json' or n=='dev.json' for n in names)
        manifest=json.loads(archive.read('evaluation-manifest.json'))
        assert len(manifest['checkpoints'])==4
        for name,digest in manifest['source_manifest'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest()==digest
        assert hashlib.sha256(archive.read('runtime-audit.json')).hexdigest()==manifest['runtime_audit_sha256']
    assert result['optimizer_updates_authorized']==0
    with pytest.raises(FileExistsError):
        prepare(**inputs)


@pytest.mark.parametrize('change', ['public','notdownloaded','wrongfile','extra','budget','unfrozen','protected','unverified','wrongrole'])
def test_rejects_invalid_controller_evidence(tmp_path,change):
    inputs=fixtures(tmp_path)
    if change=='public':inputs['dataset_receipt']['is_private']=False
    if change=='notdownloaded':inputs['dataset_receipt']['download_verified']=False
    if change=='wrongfile':inputs['dataset_receipt']['files_sha256']['raw-initial.json']='f'*64
    if change=='extra':inputs['dataset_receipt']['files_sha256']['extra.json']='f'*64
    if change=='budget':inputs['trainer_evidence']['budget_actual_updates']=1199
    if change=='unfrozen':inputs['trainer_evidence']['selected_configs_frozen']=False
    if change=='protected':inputs['trainer_evidence']['trainers'][0]['payload_files_sha256']['protected/final_iid.json']='f'*64
    if change=='unverified':inputs['trainer_evidence']['trainers'][0]['source_verified']=False
    if change=='wrongrole':inputs['checkpoint_index'][0]['role']='final'
    with pytest.raises(ValueError):prepare(**inputs)
    assert not inputs['output'].exists()


def test_provider_dataset_attachment_normalization_is_narrow(tmp_path):
    inputs=fixtures(tmp_path);prepare(**inputs)
    expected=json.loads((inputs['output']/'kernel-metadata.json').read_text())
    actual=copy.deepcopy(expected);actual['dataset_sources']=['owner/checkpoints']
    _metadata_matches(actual,expected)
    actual['dataset_sources']=['owner/checkpoints/2']
    with pytest.raises(ValueError):_metadata_matches(actual,expected)
    actual=copy.deepcopy(expected);actual['enable_gpu']=True
    with pytest.raises(ValueError):_metadata_matches(actual,expected)


def test_exact_frozen_legacy_baseline_exception(tmp_path,monkeypatch):
    from research.r6.cloud import evaluate_campaign as module
    inputs=fixtures(tmp_path)
    # Replace expected fingerprints with synthetic immutable fixture inventories;
    # production constants pin the real previously downloaded inventories.
    source={f'src/legacy/file{i}.py':'a'*64 for i in range(271)}
    source['src/mindscape/evaluation/worker.py']='b'*64
    old_files={f'legacy{i}.json':'c'*64 for i in range(18)}
    monkeypatch.setattr(module,'LEGACY_SOURCE_INVENTORY_SHA256',module._object_sha(source))
    monkeypatch.setattr(module,'LEGACY_BASELINE_FILES_SHA256',module._object_sha(old_files))
    trainer=inputs['trainer_evidence']['trainers'][0]
    trainer['metadata']['dataset_sources']=[module.LEGACY_BASELINE_DATASET]
    trainer['baseline_checkpoint_input']={'reference':module.LEGACY_BASELINE_DATASET+'/1',
        'is_private':True,'download_verified':True,'checkpoint_sha256':module.LEGACY_CHECKPOINT_SHA256,
        'files_sha256':old_files}
    trainer['legacy_frozen_source_sha256']=source
    trainer['payload_files_sha256'].update(source)
    prepare(**inputs)
    assert inputs['output'].exists()


@pytest.mark.parametrize('case',['extra-attachment','unproved-attachment','changed-source','new-r6-generator'])
def test_legacy_exception_cannot_admit_arbitrary_inputs(tmp_path,case):
    from research.r6.cloud import evaluate_campaign as module
    inputs=fixtures(tmp_path);trainer=inputs['trainer_evidence']['trainers'][0]
    if case=='extra-attachment':
        trainer['metadata']['dataset_sources']=[module.LEGACY_BASELINE_DATASET,'owner/extra']
    elif case=='unproved-attachment':
        trainer['metadata']['dataset_sources']=[module.LEGACY_BASELINE_DATASET]
    elif case=='changed-source':
        trainer['payload_files_sha256']['src/mindscape/evaluation/new.py']='f'*64
        trainer['legacy_frozen_source_sha256']={'src/mindscape/evaluation/new.py':'f'*64}
    else:
        trainer['payload_files_sha256']['research/r6/datasets/coding.py']='f'*64
    with pytest.raises(ValueError):prepare(**inputs)
