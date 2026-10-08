"""Trusted host worker; guest has only task files and read-only Python stdlib."""

import argparse
import resource
from pathlib import Path

import wasmtime


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runtime", required=True)
    p.add_argument("--repository", required=True)
    p.add_argument("--program", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--timeout", type=float, default=5)
    a = p.parse_args()
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024 * 1024, 8 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_CPU, (6, 7))
    runtime = Path(a.runtime).resolve()
    repo = Path(a.repository).resolve()
    out = Path(a.output).resolve()
    config = wasmtime.Config()
    config.consume_fuel = True
    engine = wasmtime.Engine(config)
    compiled = runtime / "python.cwasm"
    module = (
        wasmtime.Module.deserialize_file(engine, str(compiled))
        if compiled.exists()
        else wasmtime.Module.from_file(engine, str(runtime / "python.wasm"))
    )
    store = wasmtime.Store(engine)
    store.set_limits(memory_size=256 * 1024 * 1024, instances=4, memories=2)
    store.set_fuel(5000000000)
    wasi = wasmtime.WasiConfig()
    wasi.argv = ["python", "-B", "-c", Path(a.program).read_text()]
    wasi.env = [("PYTHONHOME", "/"), ("PYTHONPATH", "/task")]
    wasi.preopen_dir(str(runtime / "lib"), "/lib", fs_mutable=False)
    wasi.preopen_dir(str(repo), "/task", fs_mutable=False)
    wasi.stdout_file = str(out / "stdout")
    wasi.stderr_file = str(out / "stderr")
    store.set_wasi(wasi)
    linker = wasmtime.Linker(engine)
    linker.define_wasi()
    instance = linker.instantiate(store, module)
    try:
        instance.exports(store)["_start"](store)
    except wasmtime.ExitTrap as e:
        if e.code:
            raise SystemExit(e.code)
    except wasmtime.Trap as e:
        print("WASI trap:", str(e))
        raise SystemExit(124)


if __name__ == "__main__":
    main()
