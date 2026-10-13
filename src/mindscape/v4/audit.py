"""Check recorded predictions, checkpoint bindings, metrics and decision without neural calls."""

import hashlib
import json
from pathlib import Path

import numpy as np

from mindscape.v3.schema import digest

from .data import DOMAINS, tasks
from .scoring import metrics
from .verify import replay


def audit(root):
    root = Path(root)
    receipt = replay(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if digest(tasks()) != manifest["identity"]["tasks"]:
        raise ValueError("Task freeze mismatch")
    for name, sha in manifest["sources"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != sha:
            raise ValueError("Source changed: " + name)
    summary = json.loads((root / "summary.json").read_text())
    rows = [json.loads(line) for line in (root / "rows.jsonl").read_text().splitlines()]
    protocol = manifest["protocol"]
    seen = set()
    by = {}
    for fit in summary["fits"]:
        identity = fit["identity"]
        target, condition, seed = (identity[k] for k in ("target", "condition", "seed"))
        key = (target, condition, seed)
        if (
            key in seen
            or identity["rows"] != digest(rows)
            or identity["protocol"] != digest(protocol)
        ):
            raise ValueError("Fit identity mismatch")
        seen.add(key)
        by[key] = fit
        path = root / "fits" / f"{target}-{condition}-s{seed}.pt"
        if hashlib.sha256(path.read_bytes()).hexdigest() != fit["checkpoint_sha256"]:
            raise ValueError("Checkpoint mismatch")
        if (
            fit["gradient_updates"] != 120
            or len(fit["losses"]) != 120
            or fit["candidate_presentations"] != 3840
        ):
            raise ValueError("Compute budget mismatch")
        for split in ("iid", "ood"):
            subset = [r for r in rows if r["domain"] == target and r["split"] == split]
            predictions = fit["results"][split]["predictions"]
            if len(predictions) != len(subset):
                raise ValueError("Prediction coverage")
            for row, prediction in zip(subset, predictions):
                p = prediction["probabilities"]
                if len(p) != len(row["labels"]) or not all(0 <= v <= 1 for v in p):
                    raise ValueError("Invalid probability")
                if prediction["selected"] != int(np.argmax(p)):
                    raise ValueError("Selection mismatch")
            if metrics(subset, predictions) != fit["results"][split]["metrics"]:
                raise ValueError("Metric mismatch")
    required = {
        (d, c, s)
        for d in DOMAINS
        for c in ("domain_only", "shared", "randomized_auxiliary")
        for s in protocol["seeds"]
    }
    if seen != required:
        raise ValueError("Missing fit/control")
    for domain in DOMAINS:
        for control in ("domain_only", "randomized_auxiliary"):
            gains = [
                by[domain, "shared", s]["results"]["ood"]["metrics"]["accuracy"]
                - by[domain, control, s]["results"]["ood"]["metrics"]["accuracy"]
                for s in protocol["seeds"]
            ]
            if gains != summary["gains"][domain][control]:
                raise ValueError("Transfer delta mismatch")
        for split in ("iid", "ood"):
            subset = [r for r in rows if r["domain"] == domain and r["split"] == split]
            predictions = summary["specialist"][domain][split]["predictions"]
            if len(predictions) != len(subset) or any(
                p["selected"] != int(np.argmax(r["scores"])) for p, r in zip(predictions, subset)
            ):
                raise ValueError("Specialist selection mismatch")
            if metrics(subset, predictions) != summary["specialist"][domain][split]["metrics"]:
                raise ValueError("Specialist metric mismatch")
    decision = "GO_TO_CONFIRMATION" if all(summary["gates"].values()) else "NO_GO"
    if summary["decision"] != decision:
        raise ValueError("Decision mismatch")
    receipt.update(fits=len(seen), metric_replay="PASS", decision=decision)
    (root / "audit.json").write_text(json.dumps(receipt, indent=2))
    return receipt


if __name__ == "__main__":
    import sys

    print(json.dumps(audit(sys.argv[1])))
