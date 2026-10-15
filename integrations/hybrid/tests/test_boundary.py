"""No native source or entrypoint may acquire a hybrid dependency."""
import ast
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[3]
def test_no_reverse_imports():
    root=ROOT
    for folder in ('src','scripts'):
        for p in (root/folder).rglob('*.py'):
            for n in ast.walk(ast.parse(p.read_text())):
                names=([a.name for a in n.names] if isinstance(n,ast.Import) else [n.module or ''] if isinstance(n,ast.ImportFrom) else [])
                assert not any('mindscape_delivery' in name or 'integrations.hybrid' in name for name in names),str(p)
def test_native_without_hybrid_dependencies():
    code="import sys; sys.path.insert(0,'src'); import mindscape; from mindscape.environments.multiplication.environment import MultiplicationEnvironment; from mindscape.verification.verifier import verify; e=MultiplicationEnvironment(); e.reset((19,3)); t=e.run(); assert e.get_state().answer==57; assert verify(t,57).verified; assert not any(k.startswith(('mindscape_delivery','transformers','sympy','torch')) for k in sys.modules)"
    subprocess.run([sys.executable,'-S','-c',code],cwd=ROOT,check=True,capture_output=True)
