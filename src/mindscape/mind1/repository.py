"""Bounded static analysis of explicitly submitted snapshots; no filesystem access."""
from __future__ import annotations

import ast
from pathlib import PurePosixPath

MAX_FILES = 64
MAX_FILE_BYTES = 32_000
MAX_TOTAL_BYTES = 256_000
LANGUAGES = {'.py': 'python', '.js': 'javascript', '.jsx': 'javascript',
    '.ts': 'typescript', '.tsx': 'typescript', '.rs': 'rust', '.go': 'go',
    '.java': 'java', '.c': 'c', '.h': 'c-header', '.cpp': 'cpp', '.cs': 'csharp',
    '.sql': 'sql', '.sh': 'bash', '.html': 'html', '.css': 'css', '.jl': 'julia',
    '.r': 'r', '.m': 'matlab-or-objective-c', '.cu': 'cuda', '.json': 'json',
    '.toml': 'toml', '.yaml': 'yaml', '.yml': 'yaml', '.md': 'markdown'}


def _validate(files):
    if not isinstance(files, dict) or not 1 <= len(files) <= MAX_FILES:
        raise ValueError(f'Snapshot must contain 1–{MAX_FILES} files')
    total = 0
    for path, source in files.items():
        if not isinstance(path, str) or not path or len(path) > 240:
            raise ValueError('Invalid relative snapshot path')
        if '\\' in path or ':' in path or any(ord(c) < 32 for c in path):
            raise ValueError('Snapshot paths must use relative POSIX syntax')
        parts = path.split('/')
        if path.startswith('/') or any(p in ('', '.', '..') for p in parts):
            raise ValueError('Absolute, empty or traversing paths are unsupported')
        if not isinstance(source, str):
            raise ValueError('Snapshot contents must be text')
        try:
            size = len(source.encode('utf-8'))
        except UnicodeEncodeError:
            raise ValueError('Snapshot text must be valid UTF-8') from None
        if size > MAX_FILE_BYTES:
            raise ValueError('Snapshot file exceeds byte limit')
        total += size
    if total > MAX_TOTAL_BYTES:
        raise ValueError('Snapshot exceeds total byte limit')
    return total


def _module(path):
    name = path.removesuffix('.py').replace('/', '.')
    return name.removesuffix('.__init__') if name != '__init__' else ''


def _python(path, source):
    try:
        tree = ast.parse(source, filename=path)
    except (SyntaxError, ValueError, RecursionError) as exc:
        return {'parse_status': 'error', 'error': type(exc).__name__, 'imports': [], 'symbols': []}
    imports, symbols = [], []
    module = _module(path)
    package = module if PurePosixPath(path).name == '__init__.py' else module.rpartition('.')[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend({'module': a.name, 'line': node.lineno, 'kind': 'import'} for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = node.module or ''
            if node.level:
                parts = package.split('.') if package else []
                if node.level > len(parts):
                    prefix = '<unresolved-relative>'
                else:
                    prefix = '.'.join(parts[:len(parts) - node.level + 1] + ([prefix] if prefix else []))
            imports.append({'module': prefix, 'names': [a.name for a in node.names],
                            'line': node.lineno, 'kind': 'from'})
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.append({'name': node.name, 'kind': type(node).__name__, 'line': node.lineno})
    return {'parse_status': 'parsed', 'imports': imports, 'symbols': symbols,
            'symbol_note': 'Includes nested symbols; no call graph or runtime behavior inferred'}


def analyze_repository(files: dict) -> dict:
    """Analyze only caller-supplied text. Paths are labels, never files to open."""
    total = _validate(files)
    modules = {_module(path): path for path in files if path.endswith('.py')}
    inventory, edges, languages = [], [], {}
    for path in sorted(files):
        language = LANGUAGES.get(PurePosixPath(path).suffix.lower(), 'unknown-text')
        languages[language] = languages.get(language, 0) + 1
        item = {'path': path, 'language': language,
                'bytes': len(files[path].encode('utf-8')), 'lines': len(files[path].splitlines())}
        if language == 'python':
            item.update(_python(path, files[path]))
            for dependency in item['imports']:
                name = dependency['module']
                candidates = [name]
                if dependency['kind'] == 'from':
                    candidates += [f'{name}.{part}' if name else part for part in dependency['names'] if part != '*']
                matched = sorted({modules[n] for n in candidates if n in modules})
                if matched:
                    for target in matched:
                        edges.append({'source': path, 'target': target, 'module': name,
                                      'line': dependency['line'], 'resolution': 'snapshot-static-candidate'})
                else:
                    edges.append({'source': path, 'target': None, 'module': name,
                                  'line': dependency['line'], 'resolution': 'external-or-unresolved'})
        else:
            item.update(parse_status='inventory-only', imports=[], symbols=[])
        inventory.append(item)
    return {'ok': True, 'result': {'files': inventory, 'dependency_edges': edges,
        'language_counts': languages, 'file_count': len(files), 'total_bytes': total,
        'provenance': {'engine': 'bounded-snapshot-static-analysis', 'filesystem_read': False,
                       'execution_performed': False, 'scope': 'Caller-supplied text only'},
        'limitations': ['Import edges are static candidates, not verified runtime dependencies.',
            'No code executed, packages installed, network used, or filesystem traversed.',
            'Only Python AST imports and named definitions are analyzed; other languages receive inventory.',
            'Snapshot root is the assumed import root; src layouts and dynamic imports may remain unresolved.',
            'This is not a verified architecture, complete call graph, or security audit.']}}
