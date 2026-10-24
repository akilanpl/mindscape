import math
from copy import deepcopy
import pytest
from research.r6.datasets.physics import generate, perturbations, verify, solve, features, FEATURE_NAMES, FEATURE_SCALES


def test_deterministic_and_disjoint():
    assert generate('train',17,30) == generate('train',17,30)
    sets = [{r['canonical_key'] for r in generate(s,17,60)} for s in ('train','dev','final_iid','final_ood')]
    assert all(len(s)==60 for s in sets)
    assert all(not a & b for i,a in enumerate(sets) for b in sets[i+1:])
    assert not {r['template_id'] for r in generate('train',1,20)} & {r['template_id'] for r in generate('final_ood',2,20)}


def test_verifier_features_and_units():
    for split in ('train','dev','final_iid','final_ood'):
        for row in generate(split,29,60):
            assert len(row['input'].encode()) <= 192
            assert len(row['features']) == len(FEATURE_NAMES) == len(FEATURE_SCALES) == 17
            assert row['features'] == features(row['structured_input'])
            assert all(math.isclose(a,b,abs_tol=1e-9) for a,b in zip(verify(row['structured_input']),row['target']['state']))
            changed=deepcopy(row['structured_input']); changed['mass']=0
            with pytest.raises(ValueError): solve(changed)
            changed=deepcopy(row['structured_input']); changed['units']['force']='lbf'
            with pytest.raises(ValueError): solve(changed)


def test_interventions_obey_laws():
    for row in generate('dev',43,30):
        interventions=perturbations(row)
        original=row['target']['state']; vx,vy,x,y=row['structured_input']['initial_state']
        total=sum(s['dt'] for s in row['structured_input']['stages'])
        free=[vx,vy,x+vx*total,y+vy*total]
        for name,factor in [('double_force',2),('double_mass',.5)]:
            expected=[b+factor*(a-b) for a,b in zip(original,free)]
            assert interventions[name]['target']['state'] == pytest.approx(expected)
        assert interventions['split_time_same_force']['target']['state'] == pytest.approx(original)
        assert all(p['features'] != row['features'] for p in interventions.values())
