"""Coverage-bound primary/baseline/ablation study; independent outcome scoring."""

import json
import time
from pathlib import Path

import numpy as np
import torch

from .accelerator import require_mps
from .data import BUDGETS, SEEDS, load, training_arrays
from .environment import Environment
from .evaluation import metrics, public_batch, rollout
from .planning import plan
from .schema import HISTORY, Action, Scene, digest
from .storage import RunLock, atomic_json, file_sha, source_hash
from .training import train


def read_row(dataset, record):
    with np.load(Path(dataset) / record["file"], allow_pickle=False) as data:
        row = {k: data[k].copy() for k in data.files}
    row["horizon"] = record["scene"]["horizon"]
    return row


def normalization(dataset, family):
    # Fixed250-trajectory training normalization shared by all conditions, not holdout-derived.
    arrays = training_arrays(
        dataset, family, 250 if load(dataset, False)["scope"] == "full_fixed" else 2
    )
    state = arrays["history"][:, -1, :, :4]
    active = arrays["history"][:, -1, :, 9] > 0
    return np.maximum(state[active].std(0), 0.1).astype(np.float32)


def evaluate(dataset, records, model, kind, root, identity, scale, batch_size=64):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    new = 0
    started = time.perf_counter()
    pending = []
    for record in records:
        path = root / (record["identity"] + ".json")
        if path.exists():
            saved = json.loads(path.read_text())
            trace = saved["trace"]
            if saved["trace_hash"] != digest(trace) or saved["identity"] != identity:
                raise ValueError("Saved evaluation changed")
        else:
            pending.append(record)
    for horizon in sorted({r["scene"]["horizon"] for r in pending}):
        group = [r for r in pending if r["scene"]["horizon"] == horizon]
        for start in range(0, len(group), batch_size):
            batch = group[start : start + batch_size]
            rows = [read_row(dataset, r) for r in batch]
            public = public_batch(rows)
            begin = time.perf_counter()
            prediction = rollout(model, public, kind)
            duration = time.perf_counter() - begin
            for i, (record, row) in enumerate(zip(batch, rows, strict=True)):
                pred = {k: v[i] for k, v in prediction.items()}
                score = metrics(pred, row, scale)
                from dataclasses import asdict

                from .adapter import concept_catalog, represent

                observed_record = represent(
                    {
                        "values": public["history"][i, -1],
                        "mask": public["masks"][i, -1],
                        "context": public["contexts"][i, 0],
                    }
                )
                trace = {
                    "scene_id": record["identity"],
                    "v3_observation": asdict(observed_record),
                    "concept_catalog": concept_catalog(),
                    "event_stages": [
                        "OBSERVE",
                        "REPRESENT",
                        "PREDICT_HYPOTHETICAL",
                        "INDEPENDENT_VERIFY",
                    ],
                    "family": record["scene"]["family"],
                    "split": record["scene"]["split"],
                    "ood_condition": record["scene"]["ood_condition"],
                    "horizon": horizon,
                    "metrics": score,
                    "predictions": {k: v.tolist() for k, v in pred.items()},
                    "batch_size": len(batch),
                    "shared_batch_prediction_seconds": duration,
                    "device": "mps" if model is not None else "cpu_non_neural",
                    "kind": "hypothetical_prediction_independently_scored",
                    "normalization": scale.tolist(),
                    "private_future_to_predictor": False,
                    "model_checkpoint_index_sha256": identity.get("checkpoint"),
                }
                atomic_json(
                    root / (record["identity"] + ".json"),
                    {"identity": identity, "trace_hash": digest(trace), "trace": trace},
                )
                new += 1
    atomic_json(
        root / f"invocation-{time.time_ns()}.json",
        {
            "new_scenes": new,
            "required_scenes": len(records),
            "seconds": time.perf_counter() - started,
            "identity": identity,
        },
    )
    return new


def planning_trials(dataset, records, model, kind, root, identity, seed):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for record in records:
        path = root / (record["identity"] + ".json")
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["trace_hash"] != digest(saved["trace"]) or saved["identity"] != identity:
                raise ValueError("Planning resume changed")
            continue
        scene = Scene.from_record(record["scene"])
        env = Environment(scene)
        history = []
        masks = []
        from .schema import observe

        for t in range(HISTORY):
            observed = observe(env.state(), scene, t)
            history.append(observed["values"])
            masks.append(observed["mask"])
            if t < HISTORY - 1:
                env.step(Action.from_vector(np.zeros((3, 4))))
        target = np.asarray(
            [history[-1][0, 0] + 1.0, max(0.6, history[-1][0, 1])], dtype=np.float32
        )
        proposal = plan(
            None if kind == "model_free" else model,
            np.asarray(history),
            np.asarray(masks),
            observed["context"],
            target,
            "random" if kind == "model_free" else kind,
            seed + scene.index,
        )
        if kind == "model_free":
            selected = model.choose(np.asarray(history), observed["context"], target)
            from .planning import CANDIDATES

            proposal["selected"] = selected
            vector = np.zeros((3, 4))
            vector[0, 2:] = CANDIDATES[selected]
            proposal["action"] = vector.tolist()
            proposal["policy"] = "MPS model-free supervised policy"
        contacts = False
        for t in range(24):
            action = Action.from_vector(proposal["action"] if t == 0 else np.zeros((3, 4)))
            actual = env.step(action)
            contacts |= bool(actual[0, 8])
        distance = float(np.linalg.norm(actual[0, :2] - target))
        success = distance <= 0.75
        trace = {
            "scene_id": scene.identity,
            "family": scene.family,
            "split": scene.split,
            "condition": kind,
            "proposal": proposal,
            "target": target.tolist(),
            "actual_terminal": actual.tolist(),
            "distance": distance,
            "goal_success": success,
            "floor_or_object_contact": contacts,
            "avoid_contact_success": success and not contacts,
            "actual_environment_steps": HISTORY - 1 + 24,
            "persistent_memory_added": False,
            "model_free_policy": "NOT RUN" if kind == "simple" else None,
        }
        atomic_json(path, {"identity": identity, "trace_hash": digest(trace), "trace": trace})


def run(dataset, output, updates=400, batch_size=64, pilot=False):
    require_mps()
    manifest = load(dataset)
    if not pilot and manifest["scope"] != "full_fixed":
        raise ValueError("Full study needs complete frozen dataset")
    identity = {
        "schema": "mindscape.v3.physics.study/1",
        "dataset": manifest["hash"],
        "source": source_hash(),
        "torch": torch.__version__,
        "device": "mps",
        "precision": "float32",
        "updates": updates,
        "batch_size": batch_size,
        "scope": "pilot" if pilot else "full_fixed",
        "seeds": [11] if pilot else list(SEEDS),
        "budgets": [2] if pilot else list(BUDGETS),
        "protocol": "configs/v3_physics/protocol.json",
        "primary_required": 4 if pilot else 6000,
    }
    identity["protocol_sha256"] = file_sha(identity["protocol"])
    lock = RunLock(output, identity)
    try:
        for family in ("motion", "interaction"):
            scale = normalization(dataset, family)
            for seed in identity["seeds"]:
                for budget in identity["budgets"]:
                    variants = (
                        ("mindscape", "matched")
                        if budget != (2 if pilot else 250)
                        else (
                            "mindscape",
                            "matched",
                            "no_relationships",
                            "no_concepts",
                            "no_memory",
                            "no_history",
                            "no_feedback",
                        )
                    )
                    for variant in variants:
                        fit_identity = {
                            **identity,
                            "family": family,
                            "seed": seed,
                            "budget": budget,
                            "variant": variant,
                        }
                        folder = Path(output) / f"{family}/s{seed}/n{budget}/{variant}"
                        if pilot:
                            _, _partial = train(
                                dataset,
                                folder,
                                fit_identity,
                                updates,
                                batch_size,
                                stop_after=max(1, updates // 2),
                            )
                        model, receipt = train(dataset, folder, fit_identity, updates, batch_size)
                        if not receipt["complete"]:
                            raise RuntimeError("Incomplete fit")
                        eval_identity = {**fit_identity, "checkpoint": receipt["checkpoint_sha256"]}
                        for split in ("iid", "ood"):
                            records = [
                                r
                                for r in manifest["records"]
                                if r["scene"]["family"] == family and r["scene"]["split"] == split
                            ]
                            evaluate(
                                dataset,
                                records,
                                model,
                                None,
                                folder / f"evaluation/{split}",
                                eval_identity,
                                scale,
                                batch_size,
                            )
                            if budget == (2 if pilot else 250):
                                for planning_kind in ("learned",):
                                    planning_trials(
                                        dataset,
                                        records[: 2 if pilot else 20],
                                        model,
                                        planning_kind,
                                        folder / f"planning/{split}",
                                        eval_identity,
                                        seed,
                                    )
                        del model
            from .policy import fit_policy, teacher_data

            policy_data = teacher_data(manifest, family, Path(output) / f"{family}/policy-teachers")
            for seed in identity["seeds"]:
                policy_identity = {
                    **identity,
                    "family": family,
                    "seed": seed,
                    "budget": 2 if pilot else 250,
                    "variant": "model_free",
                }
                policy, _policy_receipt = fit_policy(
                    policy_data, Path(output) / f"{family}/policy/s{seed}", policy_identity, updates
                )
                for split in ("iid", "ood"):
                    records = [
                        r
                        for r in manifest["records"]
                        if r["scene"]["family"] == family and r["scene"]["split"] == split
                    ]
                    planning_trials(
                        dataset,
                        records[: 2 if pilot else 20],
                        policy,
                        "model_free",
                        Path(output) / f"{family}/policy/s{seed}/planning/{split}",
                        policy_identity,
                        seed,
                    )
                del policy
            for baseline in ("persistence", "constant_velocity", "analytical_available"):
                for split in ("iid", "ood"):
                    records = [
                        r
                        for r in manifest["records"]
                        if r["scene"]["family"] == family and r["scene"]["split"] == split
                    ]
                    evaluate(
                        dataset,
                        records,
                        None,
                        baseline,
                        Path(output) / f"{family}/nonneural/{baseline}/evaluation/{split}",
                        {**identity, "family": family, "variant": baseline},
                        scale,
                        batch_size,
                    )
            for kind in ("random", "simple", "analytical"):
                for split in ("iid", "ood"):
                    records = [
                        r
                        for r in manifest["records"]
                        if r["scene"]["family"] == family and r["scene"]["split"] == split
                    ]
                    planning_trials(
                        dataset,
                        records[: 2 if pilot else 20],
                        None,
                        kind,
                        Path(output) / f"{family}/nonneural/{kind}/planning/{split}",
                        {**identity, "family": family, "variant": kind},
                        11,
                    )
        # Actual coverage checked by independent analyzer; this receipt alone is not success.
        atomic_json(
            Path(output) / "driver-complete.json",
            {
                "identity": identity,
                "driver_finished": True,
                "scientific_acceptance": "pending_independent_verification_and_analysis",
            },
        )
    finally:
        lock.close()
