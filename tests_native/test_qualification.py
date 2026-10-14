import json
from pathlib import Path
import pytest
from mindscape.r56.qualification import workload, heldout_suite, prepare, validate_scientific_gate, load_training_only
from mindscape.r56.curriculum import prepare_data


def test_full_workload_preserved():
    rows=workload()
    assert len(rows)==72
    assert sum(r['optimizer_updates'] for r in rows)==144000
    assert len({(r['seed'],r['budget'],r['control']) for r in rows})==72


def test_unseen_math_contexts_and_longer_compositions_and_physics_ood():
    train=prepare_data(17)['train'];public,targets=heldout_suite(train)
    observed={(r['problem']['operation'],r['problem']['left'],r['problem']['right'],r['problem'].get('carry_in',0)) for r in train if r['problem']['domain']=='math'}
    held=[r['problem'] for r in public['single_transitions'] if r['problem']['domain']=='math']
    assert held
    assert all((p['operation'],p['left'],p['right'],p['incoming']) not in observed for p in held)
    assert sorted({r['depth'] for r in public['compositions']})==[2,4,8,16,32]
    assert all('labels' not in r['problem'] and 'target_name' not in r['problem'] for r in public['single_transitions'])
    assert len(targets)==len(public['single_transitions'])+len(public['compositions'])


def test_proposal_is_no_go_and_training_labels_separate(tmp_path):
    manifest=prepare('.',tmp_path/'package',measure=False)
    assert manifest['scientific_decision'].startswith('NO_GO')
    with pytest.raises(PermissionError,match='NO_GO'):
        validate_scientific_gate('.',tmp_path/'package',{}, {}, {})
    rows=load_training_only(tmp_path/'package',manifest,17)
    assert all(r['split']=='train' for r in rows)
    assert 'heldout' not in (tmp_path/'package'/'train-17.json').read_text()


def test_old_package_cannot_start_optimizer_or_deserialize_labels(tmp_path,monkeypatch):
    from mindscape.r56.learning import freeze,train
    freeze('.',tmp_path/'old')
    (tmp_path/'old'/'datasets.json').write_text('DO NOT DESERIALIZE HELDOUT')
    with pytest.raises(PermissionError,match='identity'):
        train('.',tmp_path/'old',{}, {}, {},tmp_path/'weights.json')
    assert not (tmp_path/'weights.json').exists()


def test_shared_gradients_and_exact_sample_update_exposure():
    from mindscape.r56.qualification import shared_gradient_audit,update_exposures
    receipt=shared_gradient_audit()
    assert receipt['weights_unchanged'] and not receipt['transfer_demonstrated']
    assert 'concepts.weight' in receipt['shared_gradient_parameter_names']
    rows=prepare_data(17)['train']
    for budget in (32,128,512):
        receipt=update_exposures(rows,budget)
        assert receipt['total_exposures']==2000
        assert sum(receipt['per_sample_update_counts'].values())==2000


def test_paired_statistics_cluster_seeds_and_require_original_predictions():
    from mindscape.r56.qualification import paired_report
    left=[{'identity':f'id{i}','seed':s,'family':f'family{i}','metric':1.,'metric_name':'exact-accuracy','repaired':False,'original_prediction':{'digit':i}} for i in range(2) for s in (17,29,43)]
    right=[dict(r,metric=0.) for r in left]
    report=paired_report(left,right)
    assert report['independent_family_clusters']==2 and report['paired_predictions']==6
    assert report['ci95']==[1.,1.] and report['scientific_success'] is False
    with pytest.raises(ValueError):paired_report(left,[dict(r,repaired=True) for r in right])
    with pytest.raises(ValueError):paired_report(left,right[:-1])


def test_coding_evaluation_matches_observed_candidate_support():
    import torch
    from mindscape.r56.neural import R56Model
    from mindscape.r56.learning import evaluate_model
    from mindscape.r56.curriculum import generate
    model=R56Model()
    with torch.no_grad():
        model.coding_choice.weight.zero_();model.coding_choice.bias.zero_()
        model.coding_choice.bias[15]=100
        model.coding_choice.bias[1]=10
    row=next(r for r in generate('train',3) if r['problem']['domain']=='coding')
    receipt=evaluate_model(model,[row])
    assert receipt['records'][0]['original_prediction']['coding_index']==1
    assert model.predict(row['problem'])['prediction']['candidate_index']==1
