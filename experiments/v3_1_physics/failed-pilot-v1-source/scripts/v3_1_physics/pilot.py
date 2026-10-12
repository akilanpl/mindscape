"""Small native MPS development pilot; sealed evaluation is deliberately unopened."""

import json
import time
from pathlib import Path

import numpy as np
import torch
from diagnose import run as diagnose

from mindscape.v3.physics_world.accelerator import require_mps
from mindscape.v3.physics_world.data import load
from mindscape.v3.physics_world.environment import Environment
from mindscape.v3.physics_world.evaluation import metrics, public_batch
from mindscape.v3.physics_world.schema import Action, Scene, digest
from mindscape.v3.physics_world.storage import RunLock, atomic_json, file_sha
from mindscape.v3.physics_world.training import checkpoint
from mindscape.v3_1.physics.evaluation import propose, rollout
from mindscape.v3_1.physics.model import Dynamics
from mindscape.v3_1.physics.training import fit


def main(root):
    require_mps()
    started = time.perf_counter()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(Path("configs/v3_1_physics/pilot.json").read_text())
    dataset = Path("results/v3_1_physics/dataset-pilot-v1")
    manifest = load(dataset)
    frozen = json.loads(Path("experiments/v3_1_physics/freeze.json").read_text())
    for path, sha in frozen["files"].items():
        if file_sha(path) != sha:
            raise ValueError("Frozen source changed")
    assert file_sha("configs/v3_1_physics/pilot.json") == frozen["protocol"]
    assert manifest["hash"] == frozen["dataset"]
    identity = {
        "dataset": manifest["hash"],
        "protocol": digest(protocol),
        "source": digest(frozen["files"]),
        "device": "mps",
        "torch": str(torch.__version__),
        "version": protocol["version"],
    }
    lock = RunLock(root, identity)
    try:
        if not (root / "resume-test.json").exists():
            short = {**protocol, "updates": 12}
            info = {
                **identity,
                "family": "motion",
                "seed": 11,
                "variant": "candidate",
                "purpose": "multi-step-resume",
            }
            fit(dataset, root / "resume/resumed", info, short, stop_after=6)
            resumed, _ = fit(dataset, root / "resume/resumed", info, short)
            full, _ = fit(dataset, root / "resume/uninterrupted", info, short)
            difference = max(
                float((p - full.state_dict()[name]).abs().max().item())
                for name, p in resumed.state_dict().items()
            )
            assert difference <= 1e-6

            def stored(folder):
                index = json.loads((root / folder / "checkpoint-index.json").read_text())
                return torch.load(
                    root / folder / index["file"], weights_only=True, map_location="cpu"
                )

            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "safe_checkpoint_comparison", "scripts/v3_physics/checkpoint_regression.py"
            )
            comparator = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(comparator)
            equal = comparator.equal
            a = stored("resume/resumed")
            b = stored("resume/uninterrupted")
            for field in ("state", "optimizer", "step", "identity"):
                equal(a[field], b[field])
            atomic_json(
                root / "resume-test.json",
                {
                    "passed": True,
                    "max_parameter_difference": difference,
                    "optimizer_state_exact": True,
                    "device": "mps",
                    "sequence_loss": True,
                },
            )
            del resumed, full
        if not (root / "diagnosis.json").exists():
            diagnose(root / "diagnosis.json")
        audit = []
        cells = []
        plans = []
        for family in ("motion", "interaction"):
            development = []
            for r in manifest["records"]:
                if r["scene"]["family"] == family and r["scene"]["split"].startswith("development"):
                    with np.load(dataset / r["file"], allow_pickle=False) as d:
                        development.append((r, {k: d[k] for k in d.files}))
            # Entire pilot reads only train/development trajectories, never sealed files.
            for seed in protocol["seeds"]:
                models = {}
                for variant in ("candidate", "matched", "no_history", "concepts"):
                    info = {**identity, "family": family, "seed": seed, "variant": variant}
                    models[variant], arrays = fit(
                        dataset, root / f"{family}/s{seed}/{variant}", info, protocol
                    )
                memory = Dynamics("memory", seed)
                memory.load_state_dict(models["candidate"].state_dict())
                memory.memory(arrays)
                models["memory"] = memory
                checkpoint(
                    root / f"{family}/s{seed}/memory/checkpoint-index.json",
                    memory,
                    torch.optim.Adam(memory.parameters()),
                    protocol["updates"],
                    {
                        **identity,
                        "family": family,
                        "seed": seed,
                        "variant": "memory",
                        "derived_from": file_sha(
                            root / f"{family}/s{seed}/candidate/checkpoint-index.json"
                        ),
                    },
                )
                # Actual neural permutation/mask tests, not CPU substitutes.
                multi = torch.nonzero(arrays["history"][:, -1, :, 9].sum(1) >= 2)[0, 0]
                check_idx = torch.stack((torch.zeros((), device="mps", dtype=torch.long), multi))
                test = arrays["history"][check_idx]
                mask = arrays["masks"][check_idx]
                action = arrays["actions"][check_idx, 0]
                context = arrays["contexts"][check_idx, 0]
                perm = torch.tensor([2, 0, 1], device="mps")
                with torch.inference_mode():
                    for model in models.values():
                        original = model(test, mask, action, context)[0]
                        corrupted = test.clone()
                        for column in (4, 6, 7):
                            corrupted[..., column] = torch.where(
                                mask[..., column] == 0, 999, corrupted[..., column]
                            )
                        torch.testing.assert_close(
                            model(corrupted, mask, action, context)[0], original
                        )
                        permuted = model(
                            test[:, :, perm], mask[:, :, perm], action[:, perm], context
                        )[0]
                        torch.testing.assert_close(
                            permuted, original[:, perm], rtol=1e-4, atol=1e-5
                        )
                    zero = Dynamics("candidate", seed)
                    for p in zero.parameters():
                        p.zero_()
                    dv, _, _ = zero.predict(test, mask, action, context)
                    assert (
                        float(
                            (test[:, -1, :, :4] + dv - arrays["targets"][check_idx, 0])
                            .square()
                            .mean()
                        )
                        > 1e-8
                    )
                audit.append(
                    {
                        "family": family,
                        "seed": seed,
                        "permutation_consistent": True,
                        "masked_parameter_invariance": True,
                        "zero_weights_not_perfect": True,
                        "zero_training_one_step_mse": float(
                            (test[:, -1, :, :4] + dv - arrays["targets"][check_idx, 0])
                            .square()
                            .mean()
                        ),
                    }
                )
                for variant, model in models.items():
                    for horizon in protocol["evaluation_horizons"]:
                        for panel in ("development_iid", "development_stress"):
                            subset = [
                                (r, {**row, "horizon": horizon})
                                for r, row in development
                                if r["scene"]["split"] == panel
                            ]
                            predicted = rollout(model, public_batch([row for r, row in subset]))
                            values = []
                            for i, (record, row) in enumerate(subset):
                                pred = {k: v[i] for k, v in predicted.items()}
                                metric = metrics(pred, row, np.ones(4))
                                trace = {
                                    "scene_id": record["identity"],
                                    "split": panel,
                                    "family": family,
                                    "horizon": horizon,
                                    "predictions": {k: v.tolist() for k, v in pred.items()},
                                    "normalization": [1, 1, 1, 1],
                                    "metrics": metric,
                                    "device": "mps",
                                    "private_future_to_predictor": False,
                                    "persistent_memory_added": False,
                                }
                                folder = root / f"{family}/s{seed}/{variant}/h{horizon}/{panel}"
                                atomic_json(
                                    folder / f"{record['identity']}.json",
                                    {
                                        "identity": {
                                            **identity,
                                            "family": family,
                                            "seed": seed,
                                            "variant": variant,
                                        },
                                        "trace": trace,
                                        "trace_hash": digest(trace),
                                    },
                                )
                                values.append(metric)
                            cells.append(
                                {
                                    "family": family,
                                    "seed": seed,
                                    "variant": variant,
                                    "horizon": horizon,
                                    "panel": panel,
                                    "scenes": len(values),
                                    "mse": float(
                                        np.mean([v["normalized_rollout_mse"] for v in values])
                                    ),
                                    "finite": all(v["finite"] for v in values),
                                    "coverage95": float(
                                        np.mean([v["uncertainty_coverage95"] for v in values])
                                    ),
                                    "floor_penetration": float(
                                        np.mean([v["floor_penetration_mean"] for v in values])
                                    ),
                                    "momentum_mse": float(
                                        np.mean([v["momentum_total_mse"] for v in values])
                                    ),
                                }
                            )
                    if variant in ("candidate", "matched"):
                        for record, row in (
                            development[:2]
                            + [
                                (r, row)
                                for r, row in development
                                if r["scene"]["split"] == "development_stress"
                            ][:2]
                        ):
                            target = (
                                np.array([1.5, 1.0])
                                if family == "motion"
                                else np.array([1.5, 0.25])
                            )
                            proposal = propose(model, row, target)
                            env = Environment(Scene.from_record(record["scene"]))
                            for _ in range(3):
                                env.step(Action.from_vector(np.zeros((3, 4))))
                            for t in range(24):
                                actual = env.step(
                                    Action.from_vector(
                                        proposal["action"] if t == 0 else np.zeros((3, 4))
                                    )
                                )
                            distance = float(np.linalg.norm(actual[0, :2] - target))
                            trace = {
                                "scene_id": record["identity"],
                                "split": record["scene"]["split"],
                                "proposal": proposal,
                                "target": target.tolist(),
                                "actual_terminal": actual.tolist(),
                                "distance": distance,
                                "goal_success": distance <= 0.75,
                            }
                            atomic_json(
                                root
                                / f"{family}/s{seed}/{variant}/planning/{record['identity']}.json",
                                {
                                    "identity": {
                                        **identity,
                                        "family": family,
                                        "seed": seed,
                                        "variant": variant,
                                    },
                                    "trace": trace,
                                    "trace_hash": digest(trace),
                                },
                            )
                            plans.append(
                                {
                                    "family": family,
                                    "seed": seed,
                                    "variant": variant,
                                    "success": distance <= 0.75,
                                }
                            )
                del models, zero, memory
        gates = {}
        components = {}
        for family in ("motion", "interaction"):
            select = lambda variant, horizon, family=family: [
                r
                for r in cells
                if r["family"] == family
                and r["variant"] == variant
                and r["horizon"] == horizon
                and r["panel"] == "development_stress"
            ]
            mean = lambda variant, horizon, select=select: float(
                np.mean([r["mse"] for r in select(variant, horizon)])
            )
            improvement = 1 - mean("candidate", 32) / max(mean("matched", 32), 1e-12)
            growth = mean("candidate", 32) / max(mean("candidate", 8), 1e-12)
            coverage = float(np.mean([r["coverage95"] for r in select("candidate", 32)]))
            success = lambda variant, family=family: float(
                np.mean(
                    [
                        r["success"]
                        for r in plans
                        if r["family"] == family and r["variant"] == variant
                    ]
                )
            )
            gates[family] = {
                "reduction": improvement,
                "growth32_over8": growth,
                "error32": mean("candidate", 32),
                "coverage95": coverage,
                "planning_delta": success("candidate") - success("matched"),
            }
            gates[family]["passed"] = bool(
                improvement >= 0.1
                and growth <= 4
                and mean("candidate", 32) <= 10
                and coverage >= 0.8
                and gates[family]["planning_delta"] >= 0
            )
        for variant in ("no_history", "concepts", "memory"):
            comparisons = []
            for family in ("motion", "interaction"):
                for seed in protocol["seeds"]:
                    get = lambda name, family=family, seed=seed: next(
                        r["mse"]
                        for r in cells
                        if r["family"] == family
                        and r["seed"] == seed
                        and r["variant"] == name
                        and r["horizon"] == 32
                        and r["panel"] == "development_stress"
                    )
                    comparisons.append(
                        {"family": family, "seed": seed, "delta": get(variant) - get("candidate")}
                    )
            components[variant] = {
                "comparisons": comparisons,
                "reproducible_error_benefit": all(r["delta"] < 0 for r in comparisons),
                "retained": False,
                "reason": "Pilot exploratory; no component certified without independent validation",
            }
        elapsed = time.perf_counter() - started
        atomic_json(
            root / "analysis.json",
            {
                "native_controls": audit,
                "cells": cells,
                "planning": plans,
                "gates": gates,
                "components": components,
                "seconds": elapsed,
                "passed": all(r["passed"] for r in gates.values())
                and all(r["finite"] for r in cells)
                and elapsed <= 600,
                "sealed_evaluation": "NOT RUN",
                "scope": "development pilot; no full-scale authorization",
            },
        )
    finally:
        lock.close()


if __name__ == "__main__":
    import sys

    main(sys.argv[1])
