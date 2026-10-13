"""Independent nontraining regressions: no model/optimizer construction or updates."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import pytest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import audit_diagnostic as independent
import common
import preserve
PLAN=json.loads((HERE/'plan.json').read_text())


@pytest.fixture
def native_fixture(tmp_path):
    import tarfile
    root=HERE.parents[1]/'experiments/r5_4/engineering_v2/evidence'
    archive=root/'native-diagnostic-v1.tar.gz'
    preserve.verify_archive(archive,root/'native-diagnostic-v1-preservation.json')
    destination=tmp_path/'preserved-native'
    destination.mkdir()
    with tarfile.open(archive,'r:gz') as bundle:
        for item in bundle:
            assert item.isfile() and Path(item.name).name==item.name
            (destination/item.name).write_bytes(bundle.extractfile(item).read())
    return destination


def test_complete_native_records_independently_audit(native_fixture):
    result=independent.audit(native_fixture)
    assert result['workers']==4 and result['total_objective_backward_calls']==96
    assert result['parameter_gradient_assertions_checked']==576
    assert result['diagnosis']['cold_start_support']
    assert result['process_peak_rss_bytes']==376832000
    assert result['diagnosis']['pooled']['markov']['measured']['mean']==pytest.approx(.126500559875)


def repair_record_mirrors(path,index):
    worker=json.loads((path/f'worker-{index}.json').read_text())
    (path/f'worker-{index}.progress.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in worker['records']))
    report=json.loads((path/'REPORT.json').read_text())
    report['workers'][index]=worker
    (path/'REPORT.json').write_text(json.dumps(report))
    receipt=json.loads((path/'RECEIPT.json').read_text())
    receipt['sha256']={name:common.sha(path/name) for name in receipt['sha256']}
    (path/'RECEIPT.json').write_text(json.dumps(receipt))


@pytest.mark.parametrize('condition',['order','gradient','weights','rss','timing','duplicate_parameter'])
def test_repaired_hashes_cannot_hide_invalid_native_records(tmp_path,condition,native_fixture):
    destination=tmp_path/'native'
    shutil.copytree(native_fixture,destination)
    w=json.loads((destination/'worker-0.json').read_text())
    row=w['records'][0]
    if condition=='order':
        row['slot']=1
    elif condition=='gradient':
        row['checks']['parameters'][0]['gradient_norm']=0.0
    elif condition=='weights':
        row['checks']['parameters'][0]['weights_unchanged']=False
    elif condition=='rss':
        row['memory_after_checks']['process_peak_rss_bytes']=4294967297
    elif condition=='timing':
        row['total_seconds']+=.1
    else:
        row['checks']['parameters'][1]['name']=row['checks']['parameters'][0]['name']
    (destination/'worker-0.json').write_text(json.dumps(w))
    repair_record_mirrors(destination,0)
    with pytest.raises(ValueError):
        independent.audit(destination)


def test_hash_and_metric_failures(tmp_path,native_fixture):
    destination=tmp_path/'native'
    shutil.copytree(native_fixture,destination)
    with (destination/'worker-1.json').open('a') as stream:
        stream.write(' ')
    with pytest.raises(ValueError,match='hash'):
        independent.audit(destination)
    receipt=json.loads((destination/'RECEIPT.json').read_text())
    receipt['sha256']['worker-1.json']=common.sha(destination/'worker-1.json')
    r=json.loads((destination/'REPORT.json').read_text())
    r['diagnosis']['pooled']['markov']['measured']['mean']+=.01
    (destination/'REPORT.json').write_text(json.dumps(r))
    receipt['sha256']['REPORT.json']=common.sha(destination/'REPORT.json')
    (destination/'RECEIPT.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='numerical'):
        independent.audit(destination)


def test_preserved_archive_member_integrity():
    root=HERE.parents[1]/'experiments/r5_4/engineering_v2/evidence'
    result=preserve.verify_archive(root/'native-diagnostic-v1.tar.gz',root/'native-diagnostic-v1-preservation.json')
    assert result['members']==13 and result['status']=='VERIFIED_BYTE_IDENTICAL_NATIVE_RECORDS'


def synthetic_records(seconds):
    return [dict(pos,full_update_seconds=float(seconds)) for pos in common.rows(PLAN)]


def test_fixed_configuration_counts_and_no_fastest_selection():
    assert PLAN['configurations']==[['markov',11],['recurrent',11],['recurrent',23],['markov',23],['markov',47],['recurrent',47]]
    assert len(list(common.rows(PLAN)))*len(PLAN['configurations'])==144
    records=synthetic_records(.1)
    for r in records:
        if r['phase']=='measured' and r['trial']==7:
            r['full_update_seconds']=1.0
    result=common.decision(records,2.,300000000,PLAN)
    assert not result['timing_pass']  # retain slow tail rather than fastest .1 sample


def test_initialization_cost_and_limits_remain_binding():
    result=common.decision(synthetic_records(.6),1.,4294967296,PLAN)
    assert result['projected_fit_seconds']==pytest.approx(721.5)
    assert not result['timing_pass'] and result['memory_pass']
    assert not common.decision(synthetic_records(.1),1.,4294967297,PLAN)['memory_pass']
    with pytest.raises(ValueError):
        common.decision(synthetic_records(.1)[:-1],1.,1,PLAN)


def test_stress_batch_is_training_only_and_frozen_config_preserved():
    original=json.loads((common.REPO/'experiments/r5_4/frozen/protocol.json').read_text())
    assert original['seeds']==[11,23,47] and original['updates_per_fit']==800 and original['max_fit_seconds']==900
    assert PLAN['optimizer']==dict(original['optimizer'])
    train=json.loads((common.REPO/'experiments/r5_4/frozen/train.json').read_text())
    stress=[t for t in train if len(t['left'])==len(t['right'])==4][:32]
    assert len(stress)==32 and all(t['split']=='train' for t in stress)
    assert PLAN['projected_fit_limit_seconds']==720 and PLAN['process_rss_limit_bytes']==4294967296


def test_no_approval_no_torch_import_or_output_creation(tmp_path):
    output=tmp_path/'qualification'
    result=subprocess.run([sys.executable,str(HERE/'qualify.py'),'--output',str(output)],capture_output=True,text=True)
    assert result.returncode!=0 and 'AUTHORIZATION' in result.stderr
    assert not output.exists()


def test_no_fit_campaign_evaluation_or_implicit_execution():
    import ast
    tree=ast.parse((HERE/'qualify.py').read_text())
    for node in ast.walk(tree):
        if isinstance(node,ast.ImportFrom):
            assert not any(v in (node.module or '').split('.') for v in ('campaign','evaluate'))
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):
            assert node.func.id not in ('fit','rollout','adapter','reference')
    assert 'optimizer.step()' in (HERE/'qualify.py').read_text()  # declared scratch cost, not executed by tests
