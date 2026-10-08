"""Reproducible runtime/model acquisition; user supplies a coding-enabled venv."""

import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download

root = Path("work/coding")
root.mkdir(parents=True, exist_ok=True)
pin = json.loads(Path("configs/coding/runtime_v1.json").read_text())
archive = root / "python-wasi.zip"
if not archive.exists():
    archive.write_bytes(urllib.request.urlopen(pin["archive_url"]).read())
if hashlib.sha256(archive.read_bytes()).hexdigest() != pin["archive_sha256"]:
    raise RuntimeError("WASI archive checksum mismatch")
with zipfile.ZipFile(archive) as z:
    for name in z.namelist():
        p = Path(name)
        if p.is_absolute() or ".." in p.parts:
            raise RuntimeError("Unsafe runtime archive entry")
    z.extractall(root / "runtime")
if (
    hashlib.sha256((root / "runtime/python.wasm").read_bytes()).hexdigest()
    != pin["python_wasm_sha256"]
):
    raise RuntimeError("WASM checksum mismatch")
for model, revision in [
    ("Qwen/Qwen2.5-Coder-0.5B-Instruct", "ea3f2471cf1b1f0db85067f1ef93848e38e88c25"),
    ("Qwen/Qwen2.5-Coder-1.5B-Instruct", "2e1fd397ee46e1388853d2af2c993145b0f1098a"),
]:
    print(
        snapshot_download(
            model,
            revision=revision,
            cache_dir=root / "hf",
            allow_patterns=["*.json", "*.safetensors", "*.txt", "LICENSE", "README.md"],
        )
    )
from mindscape.coding.sandbox import WasiSandbox

print(WasiSandbox(root / "runtime").probe())
