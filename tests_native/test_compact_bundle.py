"""Nontraining extraction parity; subprocess prevents old package class caching."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
from mindscape.r56.compact_bundle import build_bundle,extract_module

ROOT=Path(__file__).resolve().parents[1]


def test_minimal_bundle_excludes_reference_and_preparation_sources(tmp_path):
    inventory=build_bundle(ROOT,tmp_path/'runtime')
    assert len(inventory)==7
    forbidden={'predict','infer_problem','verify_prediction','exact_rule','prepare','main','prepare_data','evaluate_model','heldout_suite','source_locks'}
    for name in inventory:
        text=(tmp_path/'runtime'/name).read_text()
        tree=ast.parse(text)
        assert not {n.name for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.ClassDef))}&forbidden,name
        assert not any(isinstance(n,ast.ImportFrom) and n.module in {'curriculum','evaluation','qualification'} for n in ast.walk(tree))
    import mindscape.r56.compact_bundle as bundle
    source=(ROOT/'src/mindscape/r56/neural.py').read_text()
    original=ast.parse(source);extracted=ast.parse(extract_module(source,'neural.py'))
    get_forward=lambda t:next(n for c in t.body if isinstance(c,ast.ClassDef) and c.name=='R56Model' for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
    assert ast.dump(get_forward(original))==ast.dump(get_forward(extracted))


def test_original_and_extracted_tensor_outputs_match(tmp_path):
    build_bundle(ROOT,tmp_path/'runtime')
    code='''import json,torch
from mindscape.r56.neural import R56Model
from mindscape.r56.native_domains import pack_problem
from mindscape.r56.benchmarks import matched_baseline
problems=[{'domain':'math','operation':'add_digit','left':7,'right':8,'incoming':1},
{'domain':'coding','source':'def f(x):\\n return x+1','old_name':'x','requested_name':'y','candidates':['x','y']},
{'domain':'physics','mass':2.,'velocity':[1.,2.],'force':[3.,4.],'dt':.5}]
torch.set_num_threads(1);torch.manual_seed(17);model=R56Model().eval();baseline=matched_baseline('gru').eval();rows=[]
with torch.no_grad():
 for p in problems:
  g=pack_problem(p);a=model(g,memory=None,stateless=False);b=baseline(g,memory=None,stateless=False)
  rows.append({'graph':{k:getattr(g,k).tolist() for k in ('node_features','node_types','edges','relation_types','action','concept_ids')},'model':{k:a[k].tolist() for k in ('math_digit_logits','math_carry_logits','coding_logits','physics_delta')},'baseline':{k:b[k].tolist() for k in ('math_digit_logits','math_carry_logits','coding_logits','physics_delta')}})
print(json.dumps(rows,sort_keys=True))'''
    def run(path):
        env={**os.environ,'PYTHONPATH':str(path)}
        return subprocess.check_output([sys.executable,'-c',code],env=env,cwd=tmp_path,text=True)
    assert json.loads(run(ROOT/'src'))==json.loads(run(tmp_path/'runtime/src'))
