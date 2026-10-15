import pytest
from mindscape.r57.evaluation import evaluate,analytic,paired_cluster_interval,fit_physics_baselines,physics_monotonicity


def fixtures():
    p=[{'domain':'math','operation':'add_digit','left':7,'right':8,'carry_in':1},
       {'domain':'coding','old_name':'x','source':'x=1','requested_name':'y','candidates':['z','y','q','a']},
       {'domain':'physics','mass':2.,'velocity':[0.,0.],'force':[-4.,6.],'dt':.5}]
    return [{'identity':str(i),'family':'family'+str(i),'problem':x,'labels':analytic(x)} for i,x in enumerate(p)]


def test_exact_rules_are_separate_and_raw_neural_errors_preserved():
    rows=fixtures();result=evaluate(analytic,rows)
    assert result['metrics']['math']['joint_accuracy']==1
    assert result['metrics']['coding']['by_target_position']['1']['accuracy']==1
    assert result['metrics']['physics']['mse']==0
    bad=lambda p: {'math_digit':0,'math_carry':0} if p['domain']=='math' else analytic(p)
    r=evaluate(bad,rows)
    assert r['records'][0]['prediction']=={'math_digit':0,'math_carry':0}
    assert r['records'][0]['correct'] is False and not r['records'][0]['repaired']


def test_duplicate_alias_and_reference_separation():
    rows=fixtures();second={**rows[0],'identity':'different','problem':{**rows[0]['problem'],'incoming':1}}
    with pytest.raises(ValueError,match='Duplicate'):evaluate(analytic,[rows[0],second])
    def checked(p):
        assert 'labels' not in p and 'target_name' not in p
        return analytic(p)
    evaluate(checked,fixtures())


def test_paired_requires_same_cases_and_reports_no_significance():
    a=evaluate(analytic,fixtures())['records'];b=evaluate(analytic,fixtures())['records']
    math=[x for x in a if x['domain']=='math']
    r=paired_cluster_interval(math,math,'correct',replicates=100)
    assert r['interval95']==[0,0] and 'no significance' in r['interpretation']
    with pytest.raises(ValueError):paired_cluster_interval(a,b[:-1],'correct')


def test_training_fitted_baselines_and_monotonicity():
    rows=[]
    for i in range(1,13):
        p={'domain':'physics','velocity':[0.,0.],'mass':2.,'force':[float(i),-float(i)],'dt':1.}
        rows.append({'problem':p,'labels':analytic(p)})
    predictor,meta=fit_physics_baselines(rows)
    p={'domain':'physics','velocity':[0.,0.],'mass':2.,'force':[3.,-3.],'dt':1.}
    assert predictor('linear',p)['physics_delta']==pytest.approx([1.5,-1.5])
    assert 'training labels only' in meta['fit_source']
    r=physics_monotonicity(analytic,[p]);assert r['nondecreasing_fraction']==1
    assert all(x['strict_increase'] for x in r['records'])
