"""Read-only bookkeeping engineering fixtures; no fitted weights or optimizer."""
import pytest
from mindscape.r56.compact_evaluation import build_protected_evaluation,compose_addition,repeat_physics,evaluate_records,evaluate_checkpoint


def fixture_predictor(p):
    """Exact-rule fixture only, never a product/native neural backend."""
    if p['domain']=='math':
        total=p['left']+p['right']+p.get('incoming',p.get('carry_in',0))
        return {'digit':total%10,'outgoing':total//10}
    if p['domain']=='coding':return {'candidate_index':p['candidates'].index(p['requested_name'])}
    delta=[f*p['dt']/p['mass'] for f in p['force']]
    return {'delta':delta,'velocity':[v+d for v,d in zip(p['velocity'],delta)]}


def test_composition_bookkeeping_at_all_frozen_depths():
    for depth in (2,4,8,16,32):
        p={'left':'9'*depth,'right':'1'}
        result=compose_addition(fixture_predictor,p)
        assert result['answer']=='1'+'0'*depth
        assert len(result['transitions'])==depth
        assert all(r['observation']['incoming']==1 for r in result['transitions'][1:])
    wrong=compose_addition(lambda p:{'digit':3,'outgoing':0},{'left':'99','right':'99'})
    assert wrong['answer']=='33' and wrong['transitions'][1]['observation']['incoming']==0


def test_protected_suite_canonical_incoming_and_repeated_feedback():
    p={'domain':'math','operation':'add_digit','left':2,'right':3,'incoming':1}
    protected=build_protected_evaluation([{'problem':p}],[],[])
    suite,refs=protected['heldout_suite']
    assert not any(r['problem']=={**p,'concept_ids':[0]} for r in suite['single_transitions'])
    assert len(suite['repeated_physics'])>0
    result=repeat_physics(lambda p:{'velocity':[v+1 for v in p['velocity']],'delta':[1,1]},
      {'domain':'physics','velocity':[0,0],'repeat_steps':4})
    assert result['velocity']==[4,4]
    assert result['transitions'][-1]['observation']['velocity']==[3,3]


def test_raw_wrong_predictions_and_references_preserved_no_claim():
    p={'domain':'math','operation':'add_digit','left':1,'right':1,'incoming':0}
    protected={'dev':[{'problem':p,'labels':{'math_digit':2,'math_carry':0}}],'test':[],
               'heldout_suite':[{'single_transitions':[],'compositions':[]},{}]}
    report=evaluate_records(lambda p:{'digit':9,'outgoing':1},protected)
    record=report['records'][0]
    assert record['original_prediction']=={'digit':9,'outgoing':1}
    assert record['protected_reference']=={'digit':2,'outgoing':0}
    assert record['metric']==0 and not record['repaired'] and not report['scientifically_qualified']


def test_no_optimizer_and_immutable_artifact_gate(monkeypatch,tmp_path):
    torch=pytest.importorskip('torch')
    def forbidden(*a,**k):raise AssertionError('No optimizer allowed')
    monkeypatch.setattr(torch.optim,'Adam',forbidden);monkeypatch.setattr(torch.optim,'AdamW',forbidden)
    manifest=tmp_path/'bad.json';manifest.write_text('{}')
    with pytest.raises(ValueError):evaluate_checkpoint(tmp_path,manifest,tmp_path/'protected',tmp_path/'weights','bad',17,'graph-concept-on')


def test_coding_permutations_cover_all_positions_and_encoded_overlap_report():
    p={'domain':'math','operation':'add_digit','left':2,'right':3,'incoming':1,'concept_ids':[0]}
    row={'problem':p,'labels':{'math_digit':6,'math_carry':0}}
    protected=build_protected_evaluation([row,row],[row],[row])
    suite,refs=protected['heldout_suite']
    coding=[r for r in suite['single_transitions'] if r['problem']['domain']=='coding']
    for source in {r['problem']['source'] for r in coding}:
        positions={refs[r['identity']]['candidate_index'] for r in coding if r['problem']['source']==source}
        assert positions=={0,1,2,3}
    audit=protected['encoded_overlap_audit']
    assert audit['train_duplicate_exposures']==1
    assert audit['splits']['dev']['encoded_overlap_unique_count']==1
    assert audit['splits']['test']['encoded_overlap_exposures']==1
    assert audit['policy'].startswith('report')


def test_summary_separates_domain_for_same_axis_accuracy():
    protected={'dev':[],'test':[
      {'problem':{'domain':'math'},'labels':{'math_digit':1,'math_carry':0}},
      {'problem':{'domain':'coding'},'labels':{'coding_index':0}}],
      'heldout_suite':[{'single_transitions':[],'compositions':[]},{}]}
    def fixture(p):return {'digit':1,'outgoing':0} if p['domain']=='math' else {'candidate_index':0}
    report=evaluate_records(fixture,protected)
    assert report['summaries']['math:test:exact-accuracy']['count']==1
    assert report['summaries']['coding:test:exact-accuracy']['count']==1
    assert report['summaries']['math:test:digit-accuracy']['count']==1
    assert report['summaries']['math:test:carry-accuracy']['count']==1
    assert report['records'][0]['observed_problem']=={'domain':'math'}


def test_component_depth_and_coding_position_structure_summaries():
    coding={'domain':'coding','source':'x=1','candidates':['a','b'],'requested_name':'b'}
    math={'domain':'math','operation':'add_digit','left':1,'right':1,'incoming':0}
    add={'domain':'math','operation':'add','left':'99','right':'1'}
    protected={'dev':[],'test':[
      {'problem':math,'labels':{'math_digit':2,'math_carry':0}},
      {'problem':coding,'labels':{'coding_index':1}}],
      'heldout_suite':[{'single_transitions':[],'compositions':[{'identity':'add2','axis':'chain','problem':add}]},{'add2':{'answer':'100'}}]}
    def fixture(p):
        if p['domain']=='coding':return {'candidate_index':0}
        if p.get('left')==1:return {'digit':2,'outgoing':1}
        return fixture_predictor(p)
    report=evaluate_records(fixture,protected);summary=report['summaries']
    assert summary['math:test:digit-accuracy']=={'count':1,'metric_sum':1.}
    assert summary['math:test:carry-accuracy']=={'count':1,'metric_sum':0.}
    assert summary['math:test:joint-accuracy']=={'count':1,'metric_sum':0.}
    assert summary['math:chain:depth=2:exact-accuracy']['count']==1
    assert summary['coding:test:target-position=1:exact-accuracy']['count']==1
    c=next(r for r in report['records'] if r['domain']=='coding')
    assert c['observed_problem']['source']=='x=1' and c['expected_target_position']==1
    assert len(c['observed_source_sha256'])==len(c['observed_ast_sha256'])==64
    assert summary['coding:test:source='+c['observed_source_sha256']+':exact-accuracy']['count']==1
