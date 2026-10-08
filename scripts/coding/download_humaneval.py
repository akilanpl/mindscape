import gzip
import hashlib
import json
import urllib.request
from pathlib import Path

root = Path("work/coding/humaneval")
root.mkdir(parents=True, exist_ok=True)
req = urllib.request.Request(
    "https://api.github.com/repos/openai/human-eval/commits/master",
    headers={"User-Agent": "Mindscape-research"},
)
revision = json.load(urllib.request.urlopen(req))["sha"]
url = f"https://raw.githubusercontent.com/openai/human-eval/{revision}/data/HumanEval.jsonl.gz"
payload = urllib.request.urlopen(url).read()
(root / "HumanEval.jsonl.gz").write_bytes(payload)
manifest = {
    "repository": "https://github.com/openai/human-eval",
    "revision": revision,
    "url": url,
    "sha256": hashlib.sha256(payload).hexdigest(),
    "tasks": len(gzip.decompress(payload).splitlines()),
    "license": "MIT",
}
(root / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(manifest)
