"""One native arithmetic campaign, after committed data/source freeze and qualification."""

import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

from mindscape.v3.schema import digest

from .arithmetic import run
from .controls import Memory
from .verification import Teacher, context_key, demonstration, verify


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    root = Path("results/r5_2_math/native-one-campaign")
    root.mkdir(parents=True, exist_ok=False)
    try:
        manifest = json.loads(Path("configs/r5_2_math/tasks.json").read_text())
        protocol = manifest["protocol"]
        freeze = json.loads(Path("configs/r5_2_math/freeze.json").read_text())
        for name, value in freeze["sources"].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != value:
                raise ValueError("Source freeze changed")
        for name, value in freeze["historical_pins"].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != value:
                raise ValueError("Historical evidence changed")
        for name in ("protocol.json", "tasks.json", "freeze.json"):
            p = "configs/r5_2_math/" + name
            if subprocess.check_output(["git", "show", "HEAD:" + p]) != Path(p).read_bytes():
                raise ValueError("Uncommitted freeze")
        if (
            digest(manifest["tasks"]) != manifest["task_hash"]
            or digest(protocol) != manifest["protocol_hash"]
        ):
            raise ValueError("Task freeze")
        save(root / "manifest.json", manifest)
        save(root / "freeze.json", freeze)
        train = [t for t in manifest["tasks"] if t["split"] == "train"]
        evaluation = [t for t in manifest["tasks"] if t["split"] != "train"]
        traces = [demonstration(t) for t in train]
        # No neural model construction before qualification.
        from .training import train_examples

        held = {tuple(v) for v in protocol["held_local_contexts"]}
        seq, signs, memory, sign_memory, exposure = train_examples(train, traces, held)
        save(root / "training-demonstrations.json", traces)
        save(root / "memory.json", {"local": memory, "sign": sign_memory})
        reference = []
        localmiss = 0
        total = 0
        for task in evaluation:
            result = run(task, Teacher())
            score = verify(task, result)
            if not score["strict"]:
                raise ValueError("Independent reference mismatch")
            reference.append(
                {"identity": task["identity"], "result": result, "verification": score}
            )
            if task["split"] == "ood":
                localmiss += sum(
                    context_key(e["state"]) not in memory for e in result["transitions"]
                )
                total += len(result["transitions"])
        labels = {
            op: sum(e[0][0] == i and bool(e[3]) for s in seq for e in s)
            for i, op in enumerate(("add", "sub", "mul"))
        }
        signcounts = {v: sum(s[1] == v for s in signs) for v in (0, 1)}
        q = protocol["qualification"]
        passed = (
            all(v >= q["minimum_training_local_labels_per_operation"] for v in labels.values())
            and signcounts[1] >= q["minimum_sign_positive"]
            and signcounts[0] >= q["minimum_sign_negative"]
            and localmiss / max(1, total) >= q["minimum_memory_local_miss_fraction_ood"]
        )
        qualification = {
            "decision": "DATA_READY" if passed else "DATA_NO_GO",
            "training_local_labels": labels,
            "sign_labels": signcounts,
            "training_held_context_input_exposures": exposure,
            "ood_reference_contexts": total,
            "ood_memory_misses": localmiss,
            "ood_memory_miss_fraction": localmiss / max(1, total),
            "reference_exact_tasks": len(reference),
        }
        save(root / "qualification.json", qualification)
        save(root / "reference.json", reference)
        if not passed:
            save(root / "summary.json", qualification)
        else:
            from mindscape.v3.physics_world.accelerator import require_mps

            require_mps()
            import torch

            from .model import Neural, NumericTransition
            from .training import fit

            fits = []
            records = []
            models = {}
            (root / "checkpoints").mkdir()
            for seed in protocol["optimizer_seeds"]:
                for kind in protocol["architectures"]:
                    model, receipt = fit(seq, signs, kind, seed, protocol, root / "checkpoints")
                    models[seed, kind] = model
                    fits.append(receipt)
            for seed in protocol["optimizer_seeds"]:
                base = models[seed, "recurrent"]
                with torch.device("mps"):
                    zero = NumericTransition(seed)
                with torch.no_grad():
                    for p in zero.parameters():
                        p.zero_()
                conditions = {
                    "recurrent": Neural(base),
                    "reset_hidden": Neural(models[seed, "reset_hidden"], "reset_hidden"),
                    "memory_only": Memory(memory, sign_memory),
                    "zero_weight": Neural(zero),
                    "zero_weight_memory": Memory(memory, sign_memory, Neural(zero)),
                }
                for name, predictor in conditions.items():
                    for task in evaluation:
                        result = run(task, predictor)
                        score = verify(task, result)
                        context_diagnostics = []
                        for event, expected in zip(
                            result["transitions"], score["posthoc_transitions"], strict=True
                        ):
                            key = context_key(event["state"])
                            diag = event["prediction"]
                            row = {
                                "key": key,
                                "memory_seen": key in memory,
                                "correct": expected["correct"],
                                "withheld_supervision": tuple(
                                    key.split("|")[:1] + [int(v) for v in key.split("|")[1:]]
                                )
                                in held,
                            }
                            for name, target_value in [
                                ("digit", expected["expected_digit"]),
                                ("carry", expected["expected_carry"]),
                            ]:
                                probabilities = diag.get(name + "_probabilities")
                                if probabilities is not None:
                                    order = sorted(range(10), key=lambda v: (-probabilities[v], v))
                                    row[name + "_rank"] = order.index(target_value) + 1
                                    row[name + "_top2_contains_reference"] = (
                                        target_value in order[:2]
                                    )
                            context_diagnostics.append(row)
                        score["context_diagnostics"] = context_diagnostics
                        records.append(
                            {
                                "identity": task["identity"],
                                "split": task["split"],
                                "family": task["family"],
                                "condition": name,
                                "seed": seed,
                                "result": result,
                                "verification": score,
                            }
                        )
                save(root / f"predictions-s{seed}.json", [r for r in records if r["seed"] == seed])
            save(root / "fits.json", fits)
            from .checkpoints import audit

            save(root / "checkpoint-audit.json", audit(root / "checkpoints", fits, protocol))
            from .analysis import analyze, replay

            summary = analyze(records, protocol)
            save(root / "summary.json", summary)
            save(root / "independent-replay.json", replay(manifest, records))
    except Exception as exc:  # noqa: BLE001 -- preserve failure and stop; no retry
        save(root / "failure.json", {"type": type(exc).__name__, "message": str(exc)})
        save(root / "summary.json", {"decision": "FAILED_CAMPAIGN", "failure": str(exc)})
    files = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }
    archive = root.with_suffix(".tar.gz")
    with tarfile.open(archive, "x:gz") as tar:
        for name in files:
            tar.add(root / name, arcname=name, recursive=False)
    with tarfile.open(archive) as tar:
        restored = {
            m.name: hashlib.sha256(tar.extractfile(m).read()).hexdigest() for m in tar.getmembers()
        }
    if restored != files:
        raise ValueError("Archive verification")
    save(
        root.with_suffix(".preservation.json"),
        {
            "files": files,
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "restore": "PASS",
        },
    )
    summary = json.loads((root / "summary.json").read_text())
    print(
        json.dumps(
            {
                "run": str(root.resolve()),
                "decision": summary["decision"],
                "gates": summary.get("gates"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
