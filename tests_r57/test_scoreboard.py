import hashlib,json
from pathlib import Path
import pytest
from mindscape.r57.scoreboard import verify_freeze,invariance,majority_predictor
from mindscape.r57.evaluation import analytic


def test_freeze_blocks_before_final_and_detects_data_tamper(tmp_path):
    with pytest.raises(FileNotFoundError):verify_freeze(tmp_path)
    source=tmp_path/'src';source.mkdir();(source/'model.py').write_text('frozen')
    data=tmp_path/'data-v1';data.mkdir();(data/'final.json').write_text('[]')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    (data/'manifest.json').write_text(json.dumps({'files':{'final.json':sha(data/'final.json')}}))
    freeze={'status':'CANDIDATE_FROZEN','source_root':str(source),'source_hashes':{'model.py':sha(source/'model.py')},'data_manifest_sha256':sha(data/'manifest.json')}
    (tmp_path/'candidate-freeze.json').write_text(json.dumps(freeze));verify_freeze(tmp_path)
    (data/'final.json').write_text('[1]')
    with pytest.raises(PermissionError,match='data file'):verify_freeze(tmp_path)


def test_permutation_probe_rules_and_majority_are_explicit():
    rows=[{'identity':'x','problem':{'domain':'coding','source':'x=1','old_name':'x','requested_name':'b','candidates':['a','b','c','d']},'labels':{'coding_index':1}}]
    result=invariance(analytic,rows)
    assert result['permutation_equivariant_fraction']==1 and result['candidate_alpha_invariant_fraction']==1
    assert majority_predictor(rows)(rows[0]['problem'])=={'coding_index':1}
