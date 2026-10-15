"""Build a minimal reviewed trainer runtime without evaluator/reference sources.

AST is inspected only. Selected definitions retain exact reviewed source text;
this builder never imports or executes input modules.
"""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path

SELECTIONS = {
 'neural.py': ('ModelConfig','GraphInput','OneHotEmbedding','R56Model'),
 'native_domains.py': ('DOMAIN_IDS','_finite','_digit','_incoming','_concepts','pack_problem'),
 'benchmarks.py': ('model_accounting','controlled_graph','matched_baseline_plan','observed_tokens','matched_baseline'),
 'learning.py': ('save_checkpoint','load_checkpoint'),
 'compact_gpu.py': ('SEEDS','CONDITIONS','SCOPE','canonical','digest','read_json','write_new','file_hash',
                    'runtime_locks','_trusted_config','_signed','validate_gate','domain_loss','training_loss_summary','run_fit'),
}
IMPORTS = {
 'neural.py':'from __future__ import annotations\nfrom dataclasses import dataclass\nimport torch\nfrom torch import nn\n',
 'native_domains.py':'from __future__ import annotations\nimport ast\nimport hashlib\nimport math\nimport torch\nfrom .neural import GraphInput\n',
 'benchmarks.py':'from __future__ import annotations\n',
 'learning.py':'from __future__ import annotations\nimport hashlib\nimport json\nfrom pathlib import Path\n',
 'compact_gpu.py':'from __future__ import annotations\nimport hashlib\nimport json\nimport os\nfrom pathlib import Path\nimport stat\nimport time\n',
}


def extract_module(source, filename):
    if filename not in SELECTIONS:raise ValueError('Unreviewed module selection')
    lines=source.splitlines(keepends=True);tree=ast.parse(source)
    wanted=set(SELECTIONS[filename]);found={};pieces=[]
    for node in tree.body:
        name=node.name if isinstance(node,(ast.FunctionDef,ast.ClassDef)) else (node.targets[0].id if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) else None)
        if name not in wanted:continue
        if name in found:raise ValueError('Duplicate selected definition')
        found[name]=True
        start=min([node.lineno]+[d.lineno for d in getattr(node,'decorator_list',[])])-1
        removed=set()
        if filename=='neural.py' and name=='R56Model':
            for method in node.body:
                if isinstance(method,ast.FunctionDef) and method.name=='predict':
                    removed.update(range(method.lineno-1,method.end_lineno))
        pieces.append(''.join(lines[i] for i in range(start,node.end_lineno) if i not in removed).rstrip()+'\n')
    if wanted-set(found):raise ValueError('Missing reviewed definitions: '+','.join(sorted(wanted-set(found))))
    result='"""Extracted reviewed training runtime; no evaluator or curriculum."""\n'+IMPORTS[filename]+'\n\n'.join(pieces)
    ast.parse(result)
    return result


def runtime_inventory(root):
    root=Path(root)
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*.py')) if '__pycache__' not in p.parts}


def build_runtime(root,destination):
    """New directory only; inventory/source hashes must be frozen before approval."""
    root=Path(root).resolve();destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    source=root/'src/mindscape/r56';target=destination/'src/mindscape/r56';target.mkdir(parents=True)
    source_hashes={}
    for name in SELECTIONS:
        raw=(source/name).read_bytes();source_hashes['src/mindscape/r56/'+name]=hashlib.sha256(raw).hexdigest()
        (target/name).write_text(extract_module(raw.decode(),name))
    for name in ('src/mindscape/__init__.py','src/mindscape/r56/__init__.py'):
        (destination/name).write_text('"""Minimal frozen trainer package."""\n')
    receipt={'format':'compact-runtime-extraction-v1','source_sha256':source_hashes,
             'runtime_source_locks':runtime_inventory(destination),
             'selection':{k:list(v) for k,v in SELECTIONS.items()},
             'excluded':['heldout generators','verifiers','evaluation','curriculum','preparation','CLI'],
             'scientific_qualification':False,'training_authorized':False}
    (destination/'runtime-inventory.json').write_text(json.dumps(receipt,sort_keys=True,indent=2))
    return receipt


def build_bundle(root,destination):
    """Compatibility API: exact complete runtime hashes, no receipt inside runtime."""
    receipt=build_runtime(root,destination)
    (Path(destination)/'runtime-inventory.json').unlink()
    return runtime_inventory(destination)
