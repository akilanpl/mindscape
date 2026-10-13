"""Read-only completed-pilot diagnosis; tensor inspection only, no neural forwards/training."""

import argparse
import ast
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from mindscape.v3.schema import digest


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def counts(labels):
    return {"positive": int(sum(labels)), "negative": len(labels) - int(sum(labels))}


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def review(root, output):
    if not __debug__:
        raise RuntimeError("Verification requires assertions enabled; do not use python -O")
    output.mkdir(parents=True, exist_ok=False)
    before = {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*")) if p.is_file()}
    summary = json.loads((root / "summary.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    protocol = manifest["protocol"]
    rows = [json.loads(x) for x in (root / "rows.jsonl").read_text().splitlines()]
    tasks = json.loads((root / "tasks.json").read_text())
    assert digest(tasks) == manifest["identity"]["tasks"]
    assert digest(protocol) == manifest["identity"]["protocol"]
    assert digest(manifest["sources"]) == manifest["identity"]["source"]
    assert len(rows) == len({r["identity"] for r in rows}) == 60
    for path, expected in manifest["sources"].items():
        assert sha(Path(path)) == expected, path
    for path, expected in manifest["asset_hashes"].items():
        assert sha(Path(path)) == expected, path
    task_map = {r["identity"]: r for r in tasks}
    grouped = defaultdict(list)
    bysource = defaultdict(list)
    task_details, candidate_details, math_ood, physics_details, coding_duplicates = (
        [],
        [],
        [],
        [],
        [],
    )
    counters = defaultdict(int)
    for row in rows:
        d, split, identity = (row[k] for k in ("domain", "split", "identity"))
        index = counters[d, split]
        counters[d, split] += 1
        group = (
            "target_four"
            if split == "train" and index < 4
            else "auxiliary_only_train"
            if split == "train"
            else split
        )
        grouped[d, split].extend(row["labels"])
        grouped[d, group].extend(row["labels"]) if group != split else None
        task_details.append(
            {
                "domain": d,
                "split": split,
                "task_index": index,
                "identity": identity,
                "target_training": split == "train" and index < 4,
                **counts(row["labels"]),
            }
        )
        for i, (p, y) in enumerate(zip(row["proposals"], row["labels"])):
            if d == "mathematics":
                source = f"learned_digit_rank_{i // 2 + 1}_carry_rank_{i % 2 + 1}"
            elif d == "physics":
                source = f"learned_rollout_impulse_{p['action'][0][2]}_{p['action'][0][3]}"
            else:
                source = (
                    "unchanged_control"
                    if p["repository"] == row["public"]["repository"]
                    else f"qwen_candidate_{i + 1}"
                )
            bysource[d, split, source].append(y)
            candidate_details.append(
                {
                    "domain": d,
                    "split": split,
                    "task_index": index,
                    "identity": identity,
                    "candidate": i,
                    "source": source,
                    "positive": int(y),
                    "negative": 1 - int(y),
                    "specialist_score": row["scores"][i],
                }
            )
        if d == "mathematics" and split == "ood":
            p = row["public"]
            raw = (
                p["left"] + p["right"] + p["incoming"]
                if p["operation"] == "add"
                else p["left"] - p["right"] - p["incoming"]
            )
            expected_digit = raw % 10
            expected_carry = raw // 10 if p["operation"] == "add" else int(raw < 0)
            math_ood.append(
                {
                    "identity": identity,
                    **p,
                    "expected_digit": expected_digit,
                    "expected_carry": expected_carry,
                    "offered_digits": sorted({p["claim"]["next_digit"] for p in row["proposals"]}),
                    "offered_carries": sorted({p["claim"]["next_carry"] for p in row["proposals"]}),
                    "solvable": any(row["labels"]),
                }
            )
        if d == "physics":
            physics_details.append(
                {
                    "identity": identity,
                    "split": split,
                    "task_index": index,
                    "target": row["public"]["target"],
                    "horizon": row["public"]["horizon"],
                    "min_distance": min(o["distance"] for o in row["outcomes"]),
                    "success_indices": [i for i, y in enumerate(row["labels"]) if y],
                    "distances": [o["distance"] for o in row["outcomes"]],
                }
            )
        if d == "coding":
            fingerprints = [
                digest(
                    {
                        k: ast.dump(ast.parse(v), include_attributes=False)
                        for k, v in p["repository"].items()
                    }
                )
                for p in row["proposals"]
            ]
            if len(set(fingerprints)) < len(fingerprints):
                coding_duplicates.append(identity)
        assert (
            row["domain"] == task_map[identity]["domain"]
            and row["split"] == task_map[identity]["split"]
        )
    write_csv(output / "labels-by-task.csv", task_details)
    write_csv(output / "labels-by-candidate.csv", candidate_details)
    write_csv(
        output / "labels-by-source.csv",
        [
            {"domain": d, "split": s, "source": source, **counts(y)}
            for (d, s, source), y in bysource.items()
        ],
    )
    write_csv(
        output / "labels-by-split.csv",
        [{"domain": d, "split": s, **counts(y)} for (d, s), y in grouped.items()],
    )
    checkpoints, prediction_metrics, train_gates = [], [], []
    loaded = {}
    for fit in summary["fits"]:
        identity = fit["identity"]
        d, c, seed = (identity[k] for k in ("target", "condition", "seed"))
        path = root / "fits" / f"{d}-{c}-s{seed}.pt"
        assert sha(path) == fit["checkpoint_sha256"]
        assert json.loads(path.with_suffix(".json").read_text()) == fit
        saved = torch.load(path, map_location="cpu", weights_only=True)
        assert (
            saved["identity"] == identity
            and saved["step"] == 120
            and saved["losses"] == fit["losses"]
        )
        assert identity["rows"] == digest(rows) and identity["protocol"] == digest(protocol)
        assert len(saved["losses"]) == 120 and np.isfinite(saved["losses"]).all()
        model = saved["model"]
        optimizer = saved["optimizer"]
        param_ids = optimizer["param_groups"][0]["params"]
        assert sum(v.numel() for v in model.values()) == fit["parameters"] == 4865
        assert all(torch.isfinite(v).all().item() for v in model.values())
        steps = {}
        for (name, tensor), index in zip(model.items(), param_ids, strict=True):
            trained = name.startswith(("shared.", "encoders." + d + ".")) or c != "domain_only"
            if not trained:
                assert index not in optimizer["state"]
                continue
            state = optimizer["state"][index]
            expected_step = (
                120
                if name.startswith("shared.") or c == "domain_only"
                else 60
                if name.startswith("encoders." + d + ".")
                else 30
            )
            assert state["step"].item() == expected_step
            steps[name] = expected_step
            for key in ("exp_avg", "exp_avg_sq"):
                assert state[key].shape == tensor.shape and torch.isfinite(state[key]).all().item()
        checkpoints.append(
            {
                "target": d,
                "condition": c,
                "seed": seed,
                "sha256": sha(path),
                "parameters": 4865,
                "optimizer_steps": steps,
            }
        )
        loaded[d, c, seed] = model
        train = [r for r in rows if r["domain"] == d and r["split"] == "train"][:4]
        labels = [y for r in train for y in r["labels"]]
        constant = float(np.var(labels))
        assert np.isclose(
            constant - fit["final_target_training_brier"],
            fit["train_brier_gain_over_constant"],
            atol=1e-14,
        )
        assert np.isclose(
            fit["initial_target_training_brier"] - fit["final_target_training_brier"],
            fit["train_brier_improvement"],
            atol=1e-14,
        )
        train_gates.append(
            {
                "domain": d,
                "condition": c,
                "seed": seed,
                **counts(labels),
                "constant_brier": constant,
                "reported_final_brier": fit["final_target_training_brier"],
                "gain_over_constant": fit["train_brier_gain_over_constant"],
                "distinction_gate_pass": fit["train_brier_gain_over_constant"] >= 0.01
                and fit["train_brier_improvement"] >= 0.01,
            }
        )
        for split in ("iid", "ood"):
            subset = [r for r in rows if r["domain"] == d and r["split"] == split]
            predictions = fit["results"][split]["predictions"]
            pairs = []
            errors = []
            successes = []
            pp = []
            yy = []
            for row, p in zip(subset, predictions, strict=True):
                assert p["identity"] == row["identity"]
                probs = p["probabilities"]
                labels = row["labels"]
                assert len(probs) == len(labels) and all(0 <= v <= 1 for v in probs)
                assert p["selected"] == max(range(len(probs)), key=probs.__getitem__)
                successes.append(labels[p["selected"]])
                pp.extend(probs)
                yy.extend(labels)
                errors.extend((a - b) ** 2 for a, b in zip(probs, labels))
                pairs.extend(
                    float(probs[i] > probs[j]) + 0.5 * float(probs[i] == probs[j])
                    for i, y in enumerate(labels)
                    if y
                    for j, n in enumerate(labels)
                    if not n
                )
            ece = 0
            for bin_index in range(5):
                indices = [i for i, v in enumerate(pp) if min(int(v * 5), 4) == bin_index]
                if indices:
                    ece += (
                        len(indices)
                        / len(pp)
                        * abs(sum(pp[i] - yy[i] for i in indices) / len(indices))
                    )
            stored = fit["results"][split]["metrics"]
            assert stored["correct"] == sum(successes) and stored["accuracy"] == sum(
                successes
            ) / len(subset)
            assert np.isclose(stored["brier"], sum(errors) / len(errors), atol=1e-14)
            assert np.isclose(stored["ece_5_bins"], ece, atol=1e-14)
            prediction_metrics.append(
                {
                    "domain": d,
                    "condition": c,
                    "seed": seed,
                    "split": split,
                    "correct": int(sum(successes)),
                    "tasks": len(subset),
                    "brier": sum(errors) / len(errors),
                    "ece": ece,
                    "within_task_pairwise_accuracy": sum(pairs) / len(pairs) if pairs else None,
                }
            )
    write_csv(output / "metrics.csv", prediction_metrics)
    write_csv(output / "training-distinction-gates.csv", train_gates)
    physics_pairs = []
    for seed in protocol["seeds"]:
        fits = {
            f["identity"]["condition"]: f
            for f in summary["fits"]
            if f["identity"]["target"] == "physics" and f["identity"]["seed"] == seed
        }
        a = fits["shared"]["results"]["ood"]["predictions"]
        b = fits["randomized_auxiliary"]["results"]["ood"]["predictions"]
        ma = loaded["physics", "shared", seed]
        mb = loaded["physics", "randomized_auxiliary", seed]
        physics_pairs.append(
            {
                "seed": seed,
                "shared_selections": [p["selected"] for p in a],
                "randomized_selections": [p["selected"] for p in b],
                "max_probability_difference": max(
                    abs(x - y)
                    for u, v in zip(a, b)
                    for x, y in zip(u["probabilities"], v["probabilities"])
                ),
                "max_weight_difference": max(float((ma[k] - mb[k]).abs().max()) for k in ma),
            }
        )
    gains = summary["gains"]
    g = protocol["gate"]
    recomputed = {
        "primary_gain": all(
            np.mean(v) * 100 >= g["primary_gain_percentage_points"]
            for v in gains["coding"].values()
        ),
        "seed_consistency": all(v > 0 for values in gains["coding"].values() for v in values),
        "no_large_negative_transfer": all(
            np.mean(gains[d]["domain_only"]) * 100
            >= -g["maximum_other_domain_regression_percentage_points"]
            for d in ("mathematics", "coding", "physics")
        ),
        "candidate_availability": all(
            sum(any(r["labels"]) for r in rows if r["domain"] == d and r["split"] == "ood")
            >= g["minimum_solvable_ood_tasks_per_domain"]
            for d in ("mathematics", "coding", "physics")
        ),
        "learns_distinctions": all(
            r["distinction_gate_pass"] for r in train_gates if r["condition"] == "shared"
        ),
        "mixed_target_training_labels": all(
            all(counts(grouped[d, "target_four"]).values())
            for d in ("mathematics", "coding", "physics")
        ),
    }
    assert (
        recomputed == summary["gates"]
        and summary["decision"] == "NO_GO"
        and not all(recomputed.values())
    )
    after = {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*")) if p.is_file()}
    assert before == after
    result = {
        "status": "PASS",
        "original_files_unchanged": len(before),
        "checkpoints": checkpoints,
        "gates_recomputed": recomputed,
        "math_ood": math_ood,
        "physics_tasks": physics_details,
        "physics_shared_randomized": physics_pairs,
        "coding_ast_duplicate_tasks": coding_duplicates,
        "limits": [
            "No neural forward or training performed. Checkpoints inspected as tensors on CPU only.",
            "Evaluation metrics independently recomputed from saved probabilities; these were not regenerated from weights.",
            "Training Brier arithmetic checked against recorded scalar receipts, not independently regenerated.",
        ],
        "original_files": before,
    }
    (output / "diagnostics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "PASS",
                "checkpoints": len(checkpoints),
                "metric_sets": len(prediction_metrics),
                "original_files_unchanged": len(before),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    review(args.run, args.output)
