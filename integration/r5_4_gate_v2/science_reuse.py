"""Explicit versioned reuse boundary: no source mutation, monkeypatch or v1 run invocation."""
import ast
import copy
import hashlib
from pathlib import Path
from evidence import REPO,GateError,require,load,sha,HERE


def scientific_suffix(source):
    tree=ast.parse(source)
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    anchor=ast.dump(ast.parse('destination = Path(destination)').body[0],include_attributes=False)
    anchors=[i for i,node in enumerate(run.body) if ast.dump(node,include_attributes=False)==anchor]
    require(len(anchors)==1,'Frozen campaign boundary incompatible')
    boundary=anchors[0]
    suffix=run.body[boundary:]
    require(not any(isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load) and n.id in ('check','authorized') for statement in suffix for n in ast.walk(statement)),
            'Engineering/human gate leaked into scientific suffix')
    return run,suffix,boundary


def reuse_descriptor():
    path=REPO/'src/mindscape/r5_4/campaign.py'
    source=path.read_text()
    run,suffix,boundary=scientific_suffix(source)
    normalized='\n'.join(ast.dump(node,include_attributes=False) for node in suffix)
    return dict(campaign_source_sha256=sha(path),scientific_suffix_ast_sha256=hashlib.sha256(normalized.encode()).hexdigest(),
                replaced_preamble_statements=boundary,original_science_first_line=suffix[0].lineno,
                original_science_last_line=suffix[-1].end_lineno,
                scope='Only authorization, source/data readiness and engineering preamble is replaced; exact frozen scientific AST is reused')


def compile_scientific_body():
    """Defines the reused function; never executes it. Called only after both future approvals."""
    descriptor=reuse_descriptor()
    require(descriptor==load(HERE/'science-equivalence.json'),'Scientific reuse boundary differs from review freeze')
    _,suffix,_=scientific_suffix((REPO/'src/mindscape/r5_4/campaign.py').read_text())
    wrapper=ast.parse('def _r54_frozen_scientific_body(destination, tasks, readiness):\n    pass\n').body[0]
    wrapper.body=copy.deepcopy(suffix)
    wrapper.end_lineno=suffix[-1].end_lineno
    # No scientific loop/optimizer/checkpoint/prediction/scoring statement is rewritten.
    module=ast.fix_missing_locations(ast.Module(body=[wrapper],type_ignores=[]))
    for a,b in zip(wrapper.body,suffix):
        require(ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False),'Scientific AST mutation')
    return compile(module,'<R54_GATE_V2_FROZEN_SCIENCE_REUSE>','exec')


def invoke_frozen_science(destination,tasks,readiness):
    # Caller must already have validated separate adoption/training approvals, source locks and native runtime.
    from authorization import validate_record,ADOPTION,TRAINING
    from datetime import datetime,timezone
    import subprocess
    from evidence import verify_sources
    require(readiness.get('method')=='R54_V2_GATE_INTEGRATION','Verified gate envelope required')
    require(readiness['qualification']['final_receipt_sha256']==load(HERE/'qualification-pin.json')['final_receipt_sha256'],'Unreviewed qualification')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    for key,purpose in [('gate_adoption',ADOPTION),('scientific_training',TRAINING)]:
        validate_record(readiness['authorizations'][key],purpose,readiness['qualification'],commit,datetime.now(timezone.utc),destination)
    verify_sources()
    import mindscape.r5_4.campaign as frozen
    require(Path(frozen.__file__).resolve()==(REPO/'src/mindscape/r5_4/campaign.py').resolve(),'Substituted scientific module')
    namespace=dict(vars(frozen))  # private copy; frozen module globals and frozen.run remain untouched
    exec(compile_scientific_body(),namespace)
    return namespace['_r54_frozen_scientific_body'](destination,tasks,readiness)
