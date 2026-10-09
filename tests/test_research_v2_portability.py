"""Corruption and stream-boundary regressions for portable evidence recovery."""
import importlib.util
import io
import json
import sys
from itertools import pairwise
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/research_v2/restore_cpu_baseline.py'
spec = importlib.util.spec_from_file_location('restore_cpu_baseline', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_multipart_stream_preserves_every_byte_across_read_boundaries(tmp_path):
    payload = bytes(range(256))*1001
    boundaries = (0, 3, 4099, 70000, len(payload))
    paths = []
    for i, (start, end) in enumerate(pairwise(boundaries)):
        p = tmp_path/str(i)
        p.write_bytes(payload[start:end])
        paths.append(p)
    with io.BufferedReader(module.PartStream(paths), buffer_size=127) as stream:
        assert stream.read() == payload


def test_corrupt_part_refuses_restore_before_destination_is_created(tmp_path, monkeypatch):
    root = tmp_path/'archive'
    root.mkdir()
    manifest = root/'SNAPSHOT_MANIFEST.json'
    manifest.write_text(json.dumps({'files': {}}))
    (root/'part').write_bytes(b'corrupt')
    (root/'ARCHIVE.json').write_text(json.dumps({
        'snapshot_manifest_sha256': module.digest(manifest),
        'parts': [{'path': 'part', 'size': 7, 'sha256': '0'*64}]}))
    dest = tmp_path/'restore'
    monkeypatch.setattr(sys, 'argv', ['restore', '--archive', str(root), '--destination', str(dest)])
    with pytest.raises(RuntimeError, match='Archive part mismatch'):
        module.main()
    assert not dest.exists()
    assert not dest.with_name(dest.name+'_snapshot').exists()


def test_invalid_compiled_cache_preserved_and_rebuilt_from_verified_wasm(tmp_path):
    import hashlib

    import wasmtime

    runtime_spec = importlib.util.spec_from_file_location('prepare_runtime', SCRIPT.with_name('prepare_cpu_runtime.py'))
    runtime_module = importlib.util.module_from_spec(runtime_spec)
    runtime_spec.loader.exec_module(runtime_module)
    wasm = bytes(wasmtime.wat2wasm('(module)'))
    (tmp_path/'python.wasm').write_bytes(wasm)
    invalid = b'incompatible-cache-test-fixture'
    (tmp_path/'python.cwasm').write_bytes(invalid)
    receipt = runtime_module.prepare(tmp_path, hashlib.sha256(wasm).hexdigest())
    assert Path(receipt['preserved_cache']).read_bytes() == invalid
    assert (tmp_path/'python.cwasm').read_bytes() != invalid
    assert receipt['wasm_sha256'] == hashlib.sha256(wasm).hexdigest()
