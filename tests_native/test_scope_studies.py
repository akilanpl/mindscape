import json
import pytest
from mindscape.r56.scope_studies import semantic_candidates,semantic_reference,training_source_vocabulary,wasi_crosscheck,language_metrics,freeze_supplement


def test_independent_reference_semantics_and_structural_holdout():
    problems,targets=semantic_candidates()
    assert [targets[r['identity']]['result'] for r in problems['train']]==[14,10,7,9]
    assert [targets[r['identity']]['result'] for r in problems['heldout']]==[15,18,29,15]
    assert not {r['structure'] for r in problems['train']}&{r['structure'] for r in problems['heldout']}
    assert all('result' not in r and 'labels' not in r for rows in problems.values() for r in rows)
    vocab=training_source_vocabulary(problems)
    assert 'total' not in vocab.tokens
    assert vocab.encode(problems['heldout'][-1]['source'])['unknown_tokens']>0


@pytest.mark.parametrize('source',["import os\nresult=1", "result=eval('1')",'result = (1).__class__','while True:\n pass','result=0\nfor i in range(100):\n result=result+i','def f(x):\n return f(x)\nresult=f(1)'])
def test_reference_rejects_unsafe_or_unbounded_programs(source):
    with pytest.raises((ValueError,RecursionError)):semantic_reference(source)


def test_wasi_unavailable_never_falls_back_to_host(tmp_path):
    result=wasi_crosscheck('result = 7',7,tmp_path/'absent-runtime')
    assert result['status']=='unavailable' and result['host_execution'] is False
    assert result['agreement'] is None


def test_wasi_transport_receives_no_expected_answer(monkeypatch):
    from types import SimpleNamespace
    import mindscape.coding.sandbox as module
    class Fixture:
        def __init__(self,*args,**kwargs):pass
        def probe(self):pass
        def execute(self,repo,source):
            assert repo=={} and source=='result = 3 + 4\nprint(result)\n'
            return SimpleNamespace(stdout='7\n',returncode=0,timeout=False)
    monkeypatch.setattr(module,'WasiSandbox',Fixture)
    assert wasi_crosscheck('result = 3 + 4',999,'fixture')['agreement'] is False


def test_language_metrics_cannot_hide_present_slot_errors_behind_nulls():
    from mindscape.r56.language_learning import build_language_dataset,training_vocabulary
    data=build_language_dataset();vocab=training_vocabulary(data)
    rows=[data['paraphrase'][0],next(r for r in data['paraphrase'] if r['intent']=='unsupported')]
    bad=[{'intent':rows[0]['intent'],'spans':{}},{'abstained':True}]
    g=language_metrics(rows,bad,vocab)['groups']['paraphrase']
    assert g['present_slot_exact_rate']==0 and g['extraction_recall']==0
    assert g['null_slot_accuracy']>0 and g['abstention_precision']==1 and g['abstention_recall']==1
    perfect=[{'intent':r['intent'],'spans':r['spans']} for r in rows]
    g=language_metrics(rows,perfect,vocab)['groups']['paraphrase']
    assert g['extraction_precision']==1 and g['extraction_recall']==1 and g['all_arguments_exact']==2
    assert g['oov_tokens']>0


def test_supplement_freeze_is_separate_no_go(tmp_path):
    m=freeze_supplement(tmp_path/'package')
    assert m['primary_campaign_fits_added']==0 and m['optimizer_updates']==0
    assert m['scientific_decision']=='NO_GO_UNREVIEWED'
    assert 'protected-references.json' in m['artifacts']
