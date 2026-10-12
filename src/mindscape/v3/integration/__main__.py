"""One native CLI, with domain-selective runs and specialist-equivalence comparison."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from mindscape.v3.physics_world.storage import atomic_json, file_sha
from mindscape.v3.schema import digest

from .assets import Assets
from .runtime import Request, Runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain", choices=("all", "mathematics", "coding", "physics"), default="all"
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    root = Path(args.output or f"results/v3_integration/native-{time.time_ns()}")
    root.mkdir(parents=True, exist_ok=False)
    frozen = json.loads(Path("configs/v3_integration/requests.json").read_text())
    if digest(frozen["requests"]) != frozen["hash"]:
        raise ValueError("Request freeze changed")
    private = json.loads(Path("configs/v3_integration/private.json").read_text())
    assets = Assets(private)
    source = digest({str(p): file_sha(p) for p in Path(__file__).parent.glob("*.py")})
    runtime = Runtime(assets.factories(), source)
    results = []
    for index, record in enumerate(frozen["requests"]):
        request = Request(**record)
        if args.domain != "all" and request.domain != args.domain:
            continue
        started = time.perf_counter()
        try:
            direct = assets.factories()[request.domain](request).direct()
            unified = runtime.run(request)
            equivalent = (
                direct.prediction == unified.prediction
                and direct.actual == unified.actual
                and direct.evaluation == unified.evaluation
            )
            # Wall time/cache metadata is not a semantic model output.
            if request.domain == "coding":
                equivalent = (
                    direct.prediction["repository"] == unified.prediction["repository"]
                    and direct.actual["repository"] == unified.actual["repository"]
                    and direct.evaluation["success"] == unified.evaluation["success"]
                )
            atomic_json(
                root / f"{index}-{request.domain}.json",
                {
                    "request": asdict(request),
                    "direct": asdict(direct),
                    "unified": asdict(unified),
                    "equivalent": equivalent,
                    "seconds": time.perf_counter() - started,
                },
            )
            results.append(
                {
                    "domain": request.domain,
                    "identity": request.identity,
                    "engineering_pass": equivalent and unified.evaluation["valid"],
                    "task_success": unified.evaluation["success"],
                    "actual": unified.actual,
                }
            )
        except (ValueError, TypeError, KeyError, RuntimeError, OSError) as error:
            atomic_json(
                root / f"{index}-failure.json",
                {"domain": request.domain, "type": type(error).__name__, "error": str(error)},
            )
            results.append(
                {"domain": request.domain, "engineering_pass": False, "blocker": str(error)}
            )
    summary = {
        "engineering_integration": "PASS"
        if all(r["engineering_pass"] for r in results)
        else "FAIL",
        "learned_cross_domain_transfer": "NOT YET ESTABLISHED",
        "results": results,
        "device": "mps",
        "asset_hashes": assets.hashes,
        "suite_hash": frozen["hash"],
        "source_hash": source,
    }
    atomic_json(root / "summary.json", summary)
    print(
        json.dumps(
            {
                "run": str(root),
                "engineering_integration": summary["engineering_integration"],
                "domains": [
                    {k: v for k, v in r.items() if k not in ("actual", "identity")} for r in results
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
