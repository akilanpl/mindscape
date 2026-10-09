"""Preserve incompatible compiled caches and rebuild from checksum-pinned WASM."""
import hashlib
import platform
from pathlib import Path

import wasmtime


def prepare(runtime, expected_wasm_sha256):
    runtime = Path(runtime)
    wasm = runtime/'python.wasm'
    if hashlib.sha256(wasm.read_bytes()).hexdigest() != expected_wasm_sha256:
        raise RuntimeError('Pinned WASM checksum mismatch')
    config = wasmtime.Config()
    config.consume_fuel = True
    engine = wasmtime.Engine(config)
    cache = runtime/'python.cwasm'
    original_sha = hashlib.sha256(cache.read_bytes()).hexdigest() if cache.exists() else None
    preserved = None
    if cache.exists():
        try:
            wasmtime.Module.deserialize_file(engine, str(cache))
        except wasmtime.WasmtimeError:
            preserved = runtime/('historical-'+original_sha+'.cwasm')
            if preserved.exists():
                raise RuntimeError('Historical cache preservation path already exists')
            cache.rename(preserved)
    if not cache.exists():
        module = wasmtime.Module.from_file(engine, str(wasm))
        cache.write_bytes(module.serialize())
    wasmtime.Module.deserialize_file(engine, str(cache))
    return {'architecture': platform.machine(), 'wasm_sha256': expected_wasm_sha256,
            'original_cache_sha256': original_sha, 'preserved_cache': str(preserved) if preserved else None,
            'active_cache_sha256': hashlib.sha256(cache.read_bytes()).hexdigest(),
            'scope': 'Execution cache only; original source and research evidence unchanged'}
