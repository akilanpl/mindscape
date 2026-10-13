"""Tensor-only checkpoint checks plus independent effect/metric replay; no neural forwards."""

import hashlib
import json
from pathlib import Path

from mindscape.v3.schema import digest
from mindscape.v5.data import graph
from mindscape.v5.execute import execute

from .scoring import baseline_predictions, gate_report, metrics


def inspect_checkpoints(root, records):
    import torch

    initial = {}
    for f in records:
        d, c, s = (f["identity"][k] for k in ("target", "condition", "seed"))
        if s in initial and initial[s] != f["initial_state_hash"]:
            raise ValueError("Unmatched initialization")
        initial[s] = f["initial_state_hash"]
        if (
            f["parameters"] != 6967
            or f["updates"] != 120
            or f["graph_presentations"] != 1920
            or len(f["losses"]) != 120
            or not f["gradient_audit_pass"]
        ):
            raise ValueError("Budget/shape/gradient audit")
        expected = {
            domain: (120 if domain == d else 0)
            if c == "domain_only"
            else (60 if domain == d else 30)
            for domain in ("mathematics", "coding", "physics")
        }
        if f["domain_updates"] != expected:
            raise ValueError("Update allocation")
        for item in f["step_parameter_changes"]:
            if any(x <= 0 for x in item["block_parameter_change_l2"].values()):
                raise ValueError("Disconnected update")
        for cp in f["checkpoints"]:
            path = Path(root) / "fits" / cp["file"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != cp["sha256"]:
                raise ValueError("Checkpoint SHA")
            saved = torch.load(path, map_location="cpu", weights_only=True)
            if (
                saved["identity"] != f["identity"]
                or saved["step"] != cp["update"]
                or saved["initial_state_hash"] != f["initial_state_hash"]
            ):
                raise ValueError("Checkpoint identity")
            if saved["losses"] != f["losses"][: cp["update"]]:
                raise ValueError("Loss history mismatch")
            if sum(x.numel() for x in saved["model"].values()) != 6967 or not all(
                torch.isfinite(x).all() for x in saved["model"].values()
            ):
                raise ValueError("Parameter tensor inventory")
            from .learning import state_hash

            if cp["update"] == 120 and state_hash(saved["model"]) != f["final_state_hash"]:
                raise ValueError("Final state hash mismatch")
            updates = saved["domain_updates"]
            optimizer = saved["optimizer"]
            ids = optimizer["param_groups"][0]["params"]
            if len(ids) != len(saved["model"]):
                raise ValueError("Optimizer parameter layout")
            for (name, tensor), index in zip(saved["model"].items(), ids, strict=True):
                block = next(
                    (
                        domain
                        for domain in ("mathematics", "coding", "physics")
                        if name.startswith(("encoders." + domain + ".", "heads." + domain + "."))
                    ),
                    None,
                )
                steps = cp["update"] if name.startswith("core.") else updates[block]
                if steps == 0:
                    if index in optimizer["state"]:
                        raise ValueError("Inactive optimizer state")
                else:
                    state = optimizer["state"][index]
                    if state["step"].item() != steps:
                        raise ValueError("Adam step count")
                    for key in ("exp_avg", "exp_avg_sq"):
                        if state[key].shape != tensor.shape or not torch.isfinite(state[key]).all():
                            raise ValueError("Adam moment mismatch")
    return {
        "checkpoints": sum(len(f["checkpoints"]) for f in records),
        "matched_initializations": len(initial),
        "optimizer_audit": "PASS",
    }


def audit(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    p = manifest["protocol"]
    summary = json.loads((root / "summary.json").read_text())
    records = summary["fits"]
    targets = json.loads((root / "composition-targets.json").read_text())
    tasks = manifest["composition_tasks"]
    if (
        len(targets) != 27
        or len(tasks) != 27
        or {r["identity"] for r in targets} != {t["identity"] for t in tasks}
    ):
        raise ValueError("Composition coverage")
    if json.loads((root / "events.json").read_text()) != [
        "preflight_pass",
        "engineering_pass",
        "27_fits_and_predictions_frozen",
        "composition_targets_executed",
        "analysis_complete",
    ]:
        raise ValueError("Prediction/target order")
    tasks_by = {t["identity"]: t for t in tasks}
    for row in targets:
        task = tasks_by[row["identity"]]
        if digest(execute(task)) != digest(row["target"]) or row["graph"] != graph(task):
            raise ValueError("Independent composition effect/graph mismatch")
    frozen = json.loads((root / "prediction-freeze.json").read_text())
    for name, value in frozen["files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != value:
            raise ValueError("Post-evaluation prediction/checkpoint change")
    checks = inspect_checkpoints(root, records)
    baselines = {}
    for d in ("mathematics", "coding", "physics"):
        selected = [r for r in targets if r["domain"] == d]
        baselines[d] = {}
        for kind in ("persistence", "local_action"):
            pp = [baseline_predictions(tasks_by[r["identity"]])[kind] for r in selected]
            mm, _ = metrics(selected, pp)
            baselines[d][kind] = mm
        pp = [
            {
                "identity": r["identity"],
                "delta": r["target"]["delta"],
                "valid_probability": float(r["target"]["valid"]),
            }
            for r in selected
        ]
        mm, _ = metrics(selected, pp)
        baselines[d]["procedural_reference"] = mm
    if baselines != summary["baselines"]:
        raise ValueError("Baseline recomputation")
    for f in records:
        if (
            f["identity"]["protocol_hash"] != digest(p)
            or f["identity"]["development_hash"] != manifest["development_hash"]
            or f["identity"]["source_freeze_hash"] != manifest["freeze_hash"]
        ):
            raise ValueError("Fit source/data/protocol binding")
        rows = [r for r in targets if r["domain"] == f["identity"]["target"]]
        mm, details = metrics(rows, f["predictions"])
        if mm != f["composition"] or details != f["per_task"]:
            raise ValueError("Metric recomputation")
        saved = json.loads(
            (
                root
                / "fits"
                / f"{f['identity']['target']}-{f['identity']['condition']}-s{f['identity']['seed']}.json"
            ).read_text()
        )
        for key, value in saved.items():
            if f[key] != value:
                raise ValueError("Pre-evaluation fit receipt changed")
    expected = gate_report(records, baselines, p)
    if any(summary[key] != value for key, value in expected.items()):
        raise ValueError("Scientific gate decision mismatch")
    return {
        "status": "PASS",
        "decision": expected["decision"],
        "independent_tasks": 27,
        "fits": 27,
        **checks,
        "recorded_metric_sets": 27,
        "neural_re_forward": False,
        "targets_never_used_for_training": True,
    }
