"""One fail-closed native pilot; no automated reruns or budget expansion."""

import argparse
import json
import subprocess
import time
import traceback
from pathlib import Path

from mindscape.v3.schema import digest

from .data import tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument(
        "--resume", help="Resume training only from a fully collected, replayed run"
    )
    args = parser.parse_args()
    if args.output and args.resume:
        parser.error("Choose output or resume")
    root = Path(args.resume or args.output or f"results/v4/native-{time.time_ns()}")
    if not args.resume:
        root.mkdir(parents=True, exist_ok=False)
    frozen = tasks()
    protocol = json.loads(Path("configs/v4/protocol.json").read_text())
    import hashlib

    files = sorted(Path("src/mindscape").rglob("*.py")) + sorted(Path("scripts/v4").glob("*"))
    source = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}
    identity = {"tasks": digest(frozen), "protocol": digest(protocol), "source": digest(source)}
    try:
        if args.resume:
            manifest = json.loads((root / "manifest.json").read_text())
            if manifest["identity"] != identity:
                raise ValueError("Source/protocol/task freeze changed")
        else:
            manifest = {
                "identity": identity,
                "protocol": protocol,
                "sources": source,
                "git_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                "neural_device": "mps",
                "scientific_status": "NOT_RUN",
            }
            (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
            (root / "tasks.json").write_text(json.dumps(frozen, indent=2))
        from mindscape.v3.integration.assets import Assets
        from mindscape.v3.integration.runtime import Request
        from mindscape.v3.physics_world.accelerator import require_mps

        require_mps()  # Never attempt CPU neural execution.
        if not args.resume:
            private = {
                Request("coding", t["public"]).identity: t["private"]
                for t in frozen
                if t["domain"] == "coding"
            }
            assets = Assets(private)
            manifest["asset_hashes"] = assets.hashes
            (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
            from .collect import collect

            rows = collect(frozen, assets, root)
            del assets
            import gc

            import torch

            gc.collect()
            torch.mps.empty_cache()
        else:
            rows = [json.loads(line) for line in (root / "rows.jsonl").read_text().splitlines()]
        from .verify import replay

        replay(root)
        from .learning import run

        summary = run(rows, protocol, root / "fits")
        (root / "summary.json").write_text(json.dumps(summary, indent=2))
        # Independent outcomes and recorded metric replay after training.
        from .audit import audit

        audit(root)
        print(
            json.dumps(
                {
                    "run": str(root.resolve()),
                    "decision": summary["decision"],
                    "gates": summary["gates"],
                }
            ),
            flush=True,
        )
    except Exception as error:
        (root / f"failure-{time.time_ns()}.json").write_text(
            json.dumps(
                {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
                indent=2,
            )
        )
        raise


if __name__ == "__main__":
    main()
