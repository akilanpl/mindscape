"""No neural execution: regenerate independent targets and recheck recorded metrics/checkpoints."""

import hashlib
import json
from pathlib import Path

import numpy as np

from mindscape.v3.schema import digest

from .data import graph
from .execute import execute


def replay(root, qualification):
    # Imported metric arithmetic uses torch definitions but executes no neural model.
    from .learning import metric

    root = Path(root)
    qualification = Path(qualification)
    manifest = json.loads((qualification / "manifest.json").read_text())
    rows = json.loads((qualification / "rows.json").read_text())
    rows += [
        json.loads((qualification / i["file"]).read_text())
        for i in json.loads((qualification / "sealed-index.json").read_text())
    ]
    tasks = {t["identity"]: t for t in manifest["tasks"]}
    if len(rows) != len(tasks) or {r["identity"] for r in rows} != set(tasks):
        raise ValueError("Task coverage mismatch")
    for row in rows:
        if (
            graph(tasks[row["identity"]]) != row["graph"]
            or digest(execute(tasks[row["identity"]])) != digest(row["target"])
        ):
            raise ValueError("Independent graph/target replay mismatch")
    summary = json.loads((root / "summary.json").read_text())
    seen = set()
    by = {}
    for fit in summary["fits"]:
        ident = fit["identity"]
        d, c, s = (ident[k] for k in ("target", "condition", "seed"))
        key = (d, c, s)
        if key in seen:
            raise ValueError("Duplicate fit")
        seen.add(key)
        by[key] = fit
        path = root / "fits" / f"{d}-{c}-s{s}.pt"
        if (
            hashlib.sha256(path.read_bytes()).hexdigest() != fit["checkpoint_sha256"]
            or fit["updates"] != 120
            or fit["graph_presentations"] != 1920
        ):
            raise ValueError("Checkpoint/budget mismatch")
        import torch

        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["identity"] != ident or saved["step"] != 120 or saved["losses"] != fit["losses"]:
            raise ValueError("Checkpoint state/receipt mismatch")
        if not all(torch.isfinite(v).all().item() for v in saved["model"].values()):
            raise ValueError("Nonfinite weights")
        selected = [r for r in rows if r["domain"] == d and r["split"] == "composition"]
        if metric(selected, fit["predictions"]) != fit["composition"]:
            raise ValueError("Recorded metric mismatch")
    protocol = manifest["protocol"]
    seeds = protocol["optimizer_seeds"]
    expected = {
        (d, c, s)
        for d in ("mathematics", "coding", "physics")
        for c in ("domain_only", "shared", "randomized_auxiliary")
        for s in seeds
    }
    if seen != expected:
        raise ValueError("Missing control/seed")
    for d in ("mathematics", "coding", "physics"):
        for c in ("domain_only", "randomized_auxiliary"):
            gains = [
                (
                    by[d, c, s]["composition"]["normalized_mse"]
                    - by[d, "shared", s]["composition"]["normalized_mse"]
                )
                / max(1e-12, by[d, c, s]["composition"]["normalized_mse"])
                for s in seeds
            ]
            if gains != summary["relative_gains"][d][c]:
                raise ValueError("Transfer comparison mismatch")
    g = protocol["scientific_gates"]
    coding = [by["coding", "shared", s] for s in seeds]
    gains = summary["relative_gains"]
    checks = {
        "primary_control_gains": all(
            np.mean(v) >= g["primary_relative_mse_improvement_vs_each_control"]
            for v in gains["coding"].values()
        ),
        "seed_consistency": all(v > 0 for values in gains["coding"].values() for v in values),
        "persistence_gain": np.mean([r["composition"]["normalized_mse"] for r in coding])
        <= (1 - g["primary_relative_improvement_over_persistence"])
        * summary["baselines"]["coding"]["persistence"]["normalized_mse"],
        "validity": all(
            r["composition"]["validity_balanced_accuracy"]
            >= g["primary_validity_balanced_accuracy"]
            for r in coding
        ),
        "no_large_negative_transfer": all(
            np.mean(gains[d]["domain_only"])
            >= -g["maximum_mean_other_domain_relative_mse_regression"]
            for d in ("mathematics", "physics")
        ),
        "finite": all(r["composition"]["finite"] for r in summary["fits"]),
    }
    if {k: bool(v) for k, v in checks.items()} != summary["gates"]:
        raise ValueError("Gate recomputation mismatch")
    decision = "GO_TO_INDEPENDENT_CONFIRMATION" if all(checks.values()) else "NO_GO"
    if decision != summary["decision"]:
        raise ValueError("Decision changed")
    return {
        "status": "PASS",
        "tasks": len(tasks),
        "fits": len(seen),
        "metric_sets": len(seen),
        "decision": decision,
        "limitations": [
            "Neural probabilities/effects not re-forwarded from checkpoint.",
            "Exact native resume not claimed; no resume path.",
        ],
    }
