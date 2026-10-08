"""Fail-closed capability sandbox. Never runs task source with host Python."""

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Execution:
    stdout: str
    stderr: str
    returncode: int
    duration: float
    timeout: bool = False


class WasiSandbox:
    def __init__(self, runtime, timeout=5):
        self.runtime = Path(runtime).resolve()
        self.timeout = timeout
        if not (self.runtime / "python.wasm").is_file():
            raise RuntimeError("WASI Python runtime required; host execution is forbidden")

    def compile(self):
        import wasmtime

        p = self.runtime / "python.cwasm"
        if p.exists():
            return
        c = wasmtime.Config()
        c.consume_fuel = True
        e = wasmtime.Engine(c)
        m = wasmtime.Module.from_file(e, str(self.runtime / "python.wasm"))
        p.write_bytes(m.serialize())

    def execute(self, repository, program):
        self.compile()
        began = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="mindscape-wasi-") as tmp:
            root = Path(tmp)
            repo = root / "repo"
            repo.mkdir()
            output = root / "output"
            output.mkdir()
            script = root / "trusted_program.py"
            script.write_text(program)
            for name, content in repository.items():
                p = Path(name)
                if p.is_absolute() or ".." in p.parts or not name.endswith(".py"):
                    raise ValueError("Unsafe repository path")
                target = repo / p
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content)
            command = [
                sys.executable,
                "-m",
                "mindscape.coding.wasi_worker",
                "--runtime",
                str(self.runtime),
                "--repository",
                str(repo),
                "--program",
                str(script),
                "--output",
                str(output),
            ]
            env = {
                "PATH": os.environ.get("PATH", ""),
                "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
            }
            timeout = False
            try:
                r = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    env=env,
                    check=False,
                )
                code = r.returncode
                hosterr = r.stderr + r.stdout
            except subprocess.TimeoutExpired:
                code = 124
                timeout = True
                hosterr = "WASI wall timeout"
            stdout = (
                (output / "stdout").read_text(errors="replace")[:64000]
                if (output / "stdout").exists()
                else ""
            )
            stderr = (
                (output / "stderr").read_text(errors="replace")[:64000]
                if (output / "stderr").exists()
                else ""
            )
            return Execution(
                stdout, stderr + hosterr, code, time.perf_counter() - began, timeout or code == 124
            )

    def probe(self):
        script = "import sys,os,json\nresults={}\nfor p in ['/Users/akilan/.codex','/etc/passwd']:\n try:open(p).read();results[p]=False\n except OSError:results[p]=True\ntry:import socket;socket.socket();results['network']=False\nexcept (ImportError,OSError):results['network']=True\ntry:os.system('echo unsafe');results['process']=False\nexcept (AttributeError,OSError):results['process']=True\nprint(json.dumps(results))"
        r = self.execute({}, script)
        if r.returncode:
            raise RuntimeError("Sandbox probe failed: " + r.stderr)
        checks = json.loads(r.stdout)
        if not all(checks.values()):
            raise RuntimeError("Capability isolation failed")
        return checks
