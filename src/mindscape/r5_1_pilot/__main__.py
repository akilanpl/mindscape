"""One authorized native run; hard lock, frozen evidence, no resume or alternate settings."""

import hashlib
import json
import subprocess
import tarfile
import time
from pathlib import Path

from mindscape.r5_1.qualification import save
from mindscape.v3.schema import digest
from mindscape.v5.data import graph
from mindscape.v5.execute import execute

from .scoring import baseline_predictions, gate_report, metrics

CONFIG = Path("configs/r5_1/learning_v2")
READINESS = Path("results/r5_1/screen-2cac9d0fcef076df")
ROOT = Path("results/r5_1_learning/native-one-pilot")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_preflight(protocol, freeze):
    for name, value in freeze["sources"].items():
        if sha(name) != value:
            raise ValueError("Executable source changed: " + name)
    for name, value in freeze["prior_evidence"].items():
        if sha(name) != value:
            raise ValueError("Prior evidence changed: " + name)
    for p in CONFIG.glob("*.json"):
        if subprocess.check_output(["git", "show", "HEAD:" + str(p)]) != p.read_bytes():
            raise ValueError("Freeze must be committed")
    original = json.loads((READINESS / "source-freeze.json").read_text())
    for name, value in original["sources"].items():
        if sha(name) != value:
            raise ValueError("Readiness original source changed")
    for name, value in original["runtime_assets"].items():
        if sha(name) != value:
            raise ValueError("Runtime changed")
    from mindscape.r5_1.replay import replay

    replay(READINESS, ROOT / "readiness-replay.json")
    engineering = json.loads((READINESS / "engineering.json").read_text())
    if (
        engineering["status"] != "PASS"
        or engineering["parameters"] != 6967
        or not engineering["weights_unchanged"]
        or engineering["optimizer_steps"] != 0
    ):
        raise ValueError("Native readiness engineering")
    if (
        protocol["fits"] != 27
        or protocol["maximum_updates_per_fit"] != 120
        or protocol["batch_size"] != 16
        or protocol["optimizer_seeds"] != [11, 23, 47]
    ):
        raise ValueError("Authorized budget changed")


def bootstrap(fits, targets):
    import numpy as np

    result = {}
    rng = np.random.default_rng(2026101701)
    for domain in ("mathematics", "coding", "physics"):
        for control in ("domain_only", "randomized_auxiliary"):
            seeds = {s: {} for s in (11, 23, 47)}
            for f in fits:
                if f["identity"]["target"] == domain and f["identity"]["condition"] in (
                    "shared",
                    control,
                ):
                    seeds[f["identity"]["seed"]][f["identity"]["condition"]] = {
                        d["identity"]: d for d in f["per_task"]
                    }
            ids = [r["identity"] for r in targets if r["domain"] == domain]
            differences = []
            for ident in ids:
                ds = []
                for s in seeds:
                    a = seeds[s][control][ident]["changed_entity_mse"]
                    b = seeds[s]["shared"][ident]["changed_entity_mse"]
                    if a is not None and b is not None:
                        ds.append(a - b)
                if ds:
                    differences.append(float(np.mean(ds)))
            if differences:
                draws = np.asarray(differences)[
                    rng.integers(len(differences), size=(2000, len(differences)))
                ].mean(1)
                result[domain + "." + control] = {
                    "paired_changed_tasks": len(differences),
                    "mean_control_minus_shared": float(np.mean(differences)),
                    "task_bootstrap_95_percentile": np.quantile(draws, [0.025, 0.975]).tolist(),
                    "optimizer_seeds_not_independent_tasks": True,
                }
            else:
                result[domain + "." + control] = {"paired_changed_tasks": 0, "interval": None}
    return result


def seal():
    files = {str(p.relative_to(ROOT)): sha(p) for p in sorted(ROOT.rglob("*")) if p.is_file()}
    archive = ROOT.with_suffix(".tar.gz")
    with tarfile.open(archive, "x:gz") as tar:
        for name in files:
            tar.add(ROOT / name, arcname=name, recursive=False)
    with tarfile.open(archive) as tar:
        restored = {
            m.name: hashlib.sha256(tar.extractfile(m).read()).hexdigest() for m in tar.getmembers()
        }
    if restored != files:
        raise ValueError("Archive roundtrip mismatch")
    save(
        ROOT.with_suffix(".preservation.json"),
        {"files": files, "archive_sha256": sha(archive), "archive_roundtrip": "PASS"},
    )


def main():
    ROOT.mkdir(parents=True, exist_ok=False)  # Unique attempt lock even if preflight fails.
    started = time.perf_counter()
    phase = "preflight"
    events = []
    try:
        protocol = json.loads((CONFIG / "protocol.json").read_text())
        freeze = json.loads((CONFIG / "freeze.json").read_text())
        save(ROOT / "authorized-protocol.json", protocol)
        save(ROOT / "source-freeze.json", freeze)
        verify_preflight(protocol, freeze)
        events.append("preflight_pass")
        manifest = json.loads((READINESS / "manifest.json").read_text())
        composition = [t for t in manifest["tasks"] if t["split"] == "composition"]
        train = [
            r
            for r in json.loads((READINESS / "rows.json").read_text())
            if r["split"] == "development"
        ]
        if len(train) != 54 or len(composition) != 27:
            raise ValueError("Dataset coverage")
        binding = json.loads((CONFIG / "data-binding.json").read_text())
        if (
            digest(train) != binding["development_hash"]
            or composition != binding["composition_tasks"]
        ):
            raise ValueError("Data identity changed")
        save(
            ROOT / "manifest.json",
            {
                "protocol": protocol,
                "protocol_hash": digest(protocol),
                "freeze_hash": sha(CONFIG / "freeze.json"),
                "development_hash": digest(train),
                "development_identities": [r["identity"] for r in train],
                "composition_tasks": composition,
                "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            },
        )
        phase = "engineering"
        from mindscape.r5_1.engineering import run

        save(ROOT / "native-engineering.json", run(manifest))
        events.append("engineering_pass")
        phase = "training"
        from .learning import fit, predict

        fits = []
        (ROOT / "fits").mkdir()
        for target in ("mathematics", "coding", "physics"):
            public_tasks = [t for t in composition if t["domain"] == target]
            for seed in protocol["optimizer_seeds"]:
                for condition in ("domain_only", "shared", "randomized_auxiliary"):
                    if len(fits) >= 27:
                        raise ValueError("Fit limit")
                    model, record = fit(
                        train,
                        target,
                        condition,
                        seed,
                        protocol,
                        sha(CONFIG / "freeze.json"),
                        ROOT / "fits",
                    )
                    predictions, elapsed = predict(model, [graph(t) for t in public_tasks])
                    record["predictions"] = [
                        {"identity": t["identity"], **p}
                        for t, p in zip(public_tasks, predictions, strict=True)
                    ]
                    record["composition_prediction_seconds"] = elapsed
                    path = ROOT / "fits" / f"{target}-{condition}-s{seed}.json"
                    save(path, record)
                    fits.append(record)
                    print(
                        json.dumps(
                            {
                                "completed_fits": len(fits),
                                "target": target,
                                "condition": condition,
                                "seed": seed,
                                "updates": 120,
                            }
                        ),
                        flush=True,
                    )
        # Seal all 27 final outputs/checkpoints before first held-out execution.
        prediction_files = {
            str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "fits").glob("*"))
        }
        save(
            ROOT / "prediction-freeze.json",
            {"files": prediction_files, "fits": 27, "composition_targets_executed": False},
        )
        events.append("27_fits_and_predictions_frozen")
        save(ROOT / "events.json", events)
        phase = "composition_execution"
        targets = []
        for task in composition:
            target = execute(task)
            if digest(execute(task)) != digest(target):
                raise ValueError("Composition target replay")
            targets.append(
                {
                    "identity": task["identity"],
                    "domain": task["domain"],
                    "split": "composition",
                    "graph": graph(task),
                    "target": target,
                }
            )
            save(ROOT / "composition-targets.json", targets)
        events.append("composition_targets_executed")
        baselines = {}
        for domain in ("mathematics", "coding", "physics"):
            selected = [r for r in targets if r["domain"] == domain]
            by = {t["identity"]: t for t in composition}
            baselines[domain] = {}
            for kind in ("persistence", "local_action"):
                predictions = [baseline_predictions(by[r["identity"]])[kind] for r in selected]
                baselines[domain][kind] = metrics(selected, predictions)[0]
            expert = [
                {
                    "identity": r["identity"],
                    "delta": r["target"]["delta"],
                    "valid_probability": float(r["target"]["valid"]),
                }
                for r in selected
            ]
            baselines[domain]["procedural_reference"] = metrics(selected, expert)[0]
        for record in fits:
            selected = [r for r in targets if r["domain"] == record["identity"]["target"]]
            record["composition"], record["per_task"] = metrics(selected, record["predictions"])
        summary = {
            **gate_report(fits, baselines, protocol),
            "fits": fits,
            "baselines": baselines,
            "uncertainty": bootstrap(fits, targets),
            "seconds_total": time.perf_counter() - started,
            "integrity": "PENDING_INDEPENDENT_AUDIT",
            "limitations": [
                "Nine composition tasks/domain; optimizer seeds reuse tasks.",
                "Changed-metric denominators exclude only zero-change targets, reported separately.",
                "B repeats target supervision more than C/R; source information differs by design.",
                "Procedural expert is perfect by construction; no superiority claim above expert.",
            ],
        }
        save(ROOT / "summary.json", summary)
        events.append("analysis_complete")
        save(ROOT / "events.json", events)
        phase = "independent_audit"
        from .audit import audit

        receipt = audit(ROOT)
        save(ROOT / "independent-replay.json", receipt)
        summary["integrity"] = "PASS"
        save(ROOT / "summary.json", summary)
    except Exception as exc:  # noqa: BLE001 -- preserve every failure and stop; never restart fitting
        save(
            ROOT / "failure.json", {"phase": phase, "type": type(exc).__name__, "message": str(exc)}
        )
        save(
            ROOT / "summary.json",
            {
                "decision": "FAILED_PILOT",
                "phase": phase,
                "failure": str(exc),
                "additional_training_authorized": False,
            },
        )
    seal()
    summary = json.loads((ROOT / "summary.json").read_text())
    print(
        json.dumps(
            {
                "run": str(ROOT.resolve()),
                "decision": summary["decision"],
                "gates": summary.get("gates"),
                "integrity": summary.get("integrity"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
