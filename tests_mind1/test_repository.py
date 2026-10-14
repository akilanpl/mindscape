import pytest
from mindscape.mind1.repository import analyze_repository


def test_cross_file_imports_symbols_and_inventory():
    result = analyze_repository({'pkg/__init__.py': '',
        'pkg/a.py': 'from .b import helper\nimport json\ndef main():\n    return helper()\n',
        'pkg/b.py': 'def helper():\n    return 1\n', 'client.ts': 'import x from "./x";'} )['result']
    assert result['language_counts'] == {'typescript': 1, 'python': 3}
    assert any(edge['source'] == 'pkg/a.py' and edge['target'] == 'pkg/b.py' for edge in result['dependency_edges'])
    assert any(edge['module'] == 'json' and edge['target'] is None for edge in result['dependency_edges'])
    assert result['provenance']['filesystem_read'] is False
    assert result['files'][0]['parse_status'] == 'inventory-only'
    assert next(f for f in result['files'] if f['path'] == 'pkg/b.py')['symbols'][0]['name'] == 'helper'


def test_from_package_module_and_syntax_error():
    result = analyze_repository({'pkg/__init__.py': '', 'pkg/b.py': 'pass',
        'main.py': 'from pkg import b', 'broken.py': 'def !!!'})['result']
    assert any(edge['target'] == 'pkg/b.py' for edge in result['dependency_edges'])
    assert next(f for f in result['files'] if f['path'] == 'broken.py')['parse_status'] == 'error'


@pytest.mark.parametrize('files', [{}, {'../secret.py': ''}, {'/secret.py': ''},
    {'a//b.py': ''}, {'a/./b.py': ''}, {'a\\b.py': ''}, {'C:secret.py': ''},
    {'ok.py': None}, {'ok.py': 'x' * 32001}, {str(i): 'x' * 32000 for i in range(9)}])
def test_snapshot_limits_and_traversal_rejection(files):
    with pytest.raises(ValueError):
        analyze_repository(files)


def test_no_code_execution_or_filesystem_read(monkeypatch):
    import builtins
    monkeypatch.setattr(builtins, 'open', lambda *a, **k: (_ for _ in ()).throw(AssertionError('file read')))
    result = analyze_repository({'payload.py': 'raise RuntimeError("must not execute")\nimport os\nos.remove("secret")'})
    assert result['ok']
    assert result['result']['provenance']['execution_performed'] is False
