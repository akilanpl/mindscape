"""Single-attempt MPS inference screen. No optimizer, training, or continuation path."""

import hashlib
import json
import random
import subprocess
import tarfile
import time
from pathlib import Path

from mindscape.v3.schema import digest
from mindscape.v4.encoding import encode
from mindscape.v4.verify import outcomes

from .freeze import MANIFEST, PROPOSAL, code_key
from .gates import decision, evaluate

FREEZE = Path("configs/v4/qualification_v1/freeze.json")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def freeze_sources():
    if FREEZE.exists():
        raise FileExistsError("Source freeze already exists")
    with tarfile.open("experiments/v4/native_v1/evidence.tar.gz") as archive:
        original = json.load(archive.extractfile("manifest.json"))
    files = sorted(Path("src/mindscape").rglob("*.py")) + sorted(Path("scripts/v4").glob("*"))
    files += [MANIFEST, PROPOSAL]
    sources = {str(p): sha(p) for p in files if p.is_file()}
    pins = dict(original["asset_hashes"])
    # Bind the actual physics checkpoint as well as its index, without loading a model.
    index = next(Path(p) for p in pins if p.endswith("checkpoint-index.json"))
    info = json.loads(index.read_text())
    pins[str(index.parent / info["file"])] = info["sha256"]
    for name, expected in pins.items():
        if sha(name) != expected:
            raise ValueError("Asset mismatch: " + name)
    save(
        FREEZE,
        {
            "sources": sources,
            "assets": pins,
            "manifest_sha256": sha(MANIFEST),
            "proposal_sha256": sha(PROPOSAL),
            "maximum_requests": 36,
            "prior_v4_commit": "0c78568da683410a46c6e9465458646d92cdee6e",
        },
    )


def normalize(task, candidates, scores):
    if (
        len(candidates) != len(scores)
        or not 1 <= len(candidates) <= {"mathematics": 4, "coding": 3, "physics": 9}[task["domain"]]
    ):
        raise ValueError("Candidate budget violated")
    groups = {}
    for i, (p, s) in enumerate(zip(candidates, scores)):
        key = code_key(p["repository"]) if task["domain"] == "coding" else digest(p)
        if key not in groups:
            groups[key] = {"proposal": p, "score": s, "source_indices": [i]}
        else:
            g = groups[key]
            g["source_indices"].append(i)
            if s > g["score"]:
                g.update(proposal=p, score=s)
    values = list(groups.values())
    if task["domain"] == "coding":
        random.Random(int(task["identity"], 16)).shuffle(values)
    return values


def verify_row(task, row, root, execute=True):
    if (
        row["identity"] != task["identity"]
        or row["domain"] != task["domain"]
        or row["split"] != task["split"]
        or row["index"] != task["index"]
    ):
        raise ValueError("Task binding")
    saved = json.loads((root / "proposals" / f"{task['identity']}.json").read_text())
    if digest(saved) != row["proposal_digest"] or any(row[k] != v for k, v in saved.items()):
        raise ValueError("Prediction receipt changed")
    raw = json.loads((root / "raw" / f"{task['identity']}.json").read_text())
    if digest(raw) != row["raw_digest"] or raw["public"] != row["public"]:
        raise ValueError("Raw generation changed")
    normalized = normalize(task, raw["candidates"], raw["scores"])
    if normalized != row["normalized"]:
        raise ValueError("Deduplication/order/score binding")
    features = [
        encode(task["domain"], row["public"], p, s) for p, s in zip(row["proposals"], row["scores"])
    ]
    if (
        features != row["features"]
        or row["proposals"] != [v["proposal"] for v in normalized]
        or row["scores"] != [v["score"] for v in normalized]
    ):
        raise ValueError("Public feature boundary")
    if execute and outcomes(task, row["proposals"]) != row["outcomes"]:
        raise ValueError("Independent replay mismatch")
    if row["labels"] != [int(o["success"]) for o in row["outcomes"]]:
        raise ValueError("Labels changed")
    if task["domain"] == "mathematics":
        target = task["scorer"]
        if row["labels"] != [
            int(
                p["claim"]["next_digit"] == target["digit"]
                and p["claim"]["next_carry"] == target["carry"]
            )
            for p in row["proposals"]
        ]:
            raise ValueError("Frozen arithmetic scorer mismatch")


def verify_prefix(tasks, rows, proposal):
    for n, row in enumerate(rows):
        if row["identity"] != tasks[n]["identity"]:
            raise ValueError("Non-prefix request order")
        if n and decision(evaluate(rows[:n], proposal), n) != "PENDING":
            raise ValueError("Inference continued after terminal decision")
    if len(rows) > 36:
        raise ValueError("Request budget")


def seal(root, summary):
    save(root / "summary.json", summary)
    files = {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*")) if p.is_file()}
    archive = root.with_suffix(".tar.gz")
    with tarfile.open(archive, "x:gz") as tar:
        for name in files:
            tar.add(root / name, arcname=name, recursive=False)
    # Compare archived bytes directly with complete original inventory; no self-including checksum.
    with tarfile.open(archive, "r:gz") as tar:
        restored = {
            m.name: hashlib.sha256(tar.extractfile(m).read()).hexdigest() for m in tar.getmembers()
        }
    if restored != files:
        raise ValueError("Archive round-trip mismatch")
    save(
        root.with_suffix(".preservation.json"),
        {"files": files, "archive_sha256": sha(archive), "archive_roundtrip": "PASS"},
    )


def main():
    frozen = json.loads(FREEZE.read_text())
    manifest = json.loads(MANIFEST.read_text())
    proposal = json.loads(PROPOSAL.read_text())
    for name, expected in frozen["sources"].items():
        if sha(name) != expected:
            raise ValueError("Frozen source changed: " + name)
    if sha(MANIFEST) != frozen["manifest_sha256"] or sha(PROPOSAL) != frozen["proposal_sha256"]:
        raise ValueError("Freeze changed")
    for path in (MANIFEST, FREEZE, PROPOSAL):
        committed = subprocess.check_output(["git", "show", "HEAD:" + str(path)])
        if committed != path.read_bytes():
            raise ValueError("Manifest/protocol must be committed before inference")
    tasks = manifest["tasks"]
    if len(tasks) != 36 or digest(tasks) != manifest["tasks_sha256"]:
        raise ValueError("Task freeze invalid")
    # Fixed unique root is also the attempt lock; no automatic retry or seed search.
    root = Path("results/v4_qualification") / ("screen-" + sha(MANIFEST)[:16])
    root.mkdir(parents=True, exist_ok=False)
    (root / "proposals").mkdir()
    (root / "raw").mkdir()
    (root / "scored").mkdir()
    save(root / "tasks.json", manifest)
    save(root / "freeze.json", frozen)
    save(root / "proposal.json", proposal)
    rows = []
    attempted = 0
    error = None
    try:
        import torch

        from mindscape.v3.integration.assets import Assets
        from mindscape.v3.integration.runtime import Request
        from mindscape.v4.collect import proposals

        for name, expected in frozen["assets"].items():
            if sha(name) != expected:
                raise ValueError("Pinned asset changed: " + name)
        private = {
            Request("coding", t["public"]).identity: t["private"]
            for t in tasks
            if t["domain"] == "coding"
        }
        assets = Assets(private)  # MPS required; no CPU fallback.
        for task in tasks:
            attempted += 1
            save(
                root / "attempt.json",
                {
                    "attempted_requests": attempted,
                    "last_identity": task["identity"],
                    "commit": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], text=True
                    ).strip(),
                },
            )
            start = time.perf_counter()
            # Strip every private/scorer/split/identity field before specialist generation.
            with torch.inference_mode():
                public, candidates, scores, provenance = proposals(
                    {"domain": task["domain"], "public": task["public"]}, assets
                )
            raw = {
                "public": public,
                "candidates": candidates,
                "scores": scores,
                "provenance": provenance,
            }
            save(root / "raw" / f"{task['identity']}.json", raw)
            normalized = normalize(task, candidates, scores)
            ps = [v["proposal"] for v in normalized]
            ss = [v["score"] for v in normalized]
            row = {k: task[k] for k in ("identity", "domain", "split", "index")}
            row.update(
                public=public,
                proposals=ps,
                scores=ss,
                normalized=normalized,
                raw_digest=digest(raw),
                features=[encode(task["domain"], public, p, s) for p, s in zip(ps, ss)],
            )
            save(root / "proposals" / f"{task['identity']}.json", row)
            row["proposal_digest"] = digest(row.copy())
            row["outcomes"] = outcomes(task, ps, assets.loaded.get("coding", [None] * 4)[3])
            row["labels"] = [int(o["success"]) for o in row["outcomes"]]
            save(root / "scored" / f"{task['identity']}.json", row)
            # Independent execution immediately, so replay failures stop before the next request.
            verify_row(task, row, root)
            row["replay"] = "PASS"
            row["seconds"] = time.perf_counter() - start
            rows.append(row)
            with (root / "rows.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            gates = evaluate(rows, proposal)
            state = decision(gates, len(rows))
            save(
                root / "progress.json", {"decision": state, "completed": len(rows), "gates": gates}
            )
            print(
                json.dumps(
                    {
                        "completed": len(rows),
                        "domain": task["domain"],
                        "split": task["split"],
                        "positive": sum(row["labels"]),
                        "negative": len(ps) - sum(row["labels"]),
                        "decision": state,
                    }
                ),
                flush=True,
            )
            if state != "PENDING":
                break
        verify_prefix(tasks, rows, proposal)
    except Exception as exc:  # noqa: BLE001 -- seal every terminal failure; never continue
        error = {"type": type(exc).__name__, "message": str(exc)}
        save(root / "failure.json", error)
    assets_unchanged = all(Path(p).is_file() and sha(p) == v for p, v in frozen["assets"].items())
    gates = evaluate(rows, proposal)
    state = "DATA_NO_GO" if error or not assets_unchanged else decision(gates, len(rows))
    if state == "PENDING":
        state = "DATA_NO_GO"
        error = {"type": "IncompleteRun", "message": "No terminal gate decision"}
    summary = {
        "decision": state,
        "completed_requests": len(rows),
        "attempted_requests": attempted,
        "unexecuted_requests": 36 - attempted,
        "gates": gates,
        "failure": error,
        "independent_replay": "PASS"
        if error is None and all(r.get("replay") == "PASS" for r in rows)
        else "INCOMPLETE_OR_FAILED",
        "candidate_outcomes": sum(len(r["labels"]) for r in rows),
        "assets_unchanged": assets_unchanged,
        "training_executed": False,
        "research_sprint": "STOPPED",
        "website_changes_authorized": False,
    }
    seal(root, summary)
    print(json.dumps({"run": str(root.resolve()), **summary}), flush=True)


def replay(root, output):
    root = Path(root)
    manifest = json.loads((root / "tasks.json").read_text())
    proposal = json.loads((root / "proposal.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    preservation = json.loads(root.with_suffix(".preservation.json").read_text())
    actual_files = {str(p.relative_to(root)): sha(p) for p in root.rglob("*") if p.is_file()}
    if (
        actual_files != preservation["files"]
        or sha(root.with_suffix(".tar.gz")) != preservation["archive_sha256"]
    ):
        raise ValueError("Evidence integrity mismatch")
    if (
        digest(proposal) != manifest["proposal_sha256"]
        or digest(manifest["tasks"]) != manifest["tasks_sha256"]
    ):
        raise ValueError("Frozen manifest/proposal mismatch")
    rows_path = root / "rows.jsonl"
    rows = (
        [json.loads(line) for line in rows_path.read_text().splitlines()]
        if rows_path.exists()
        else []
    )
    verify_prefix(manifest["tasks"], rows, proposal)
    for task, row in zip(manifest["tasks"], rows):
        verify_row(task, row, root)
    gates = evaluate(rows, proposal)
    expected = (
        "DATA_NO_GO"
        if summary["failure"] or not summary["assets_unchanged"]
        else decision(gates, len(rows))
    )
    if (
        gates != summary["gates"]
        or expected != summary["decision"]
        or len(rows) != summary["completed_requests"]
    ):
        raise ValueError("Gate/outcome summary mismatch")
    receipt = {
        "status": "PASS",
        "completed_requests": len(rows),
        "candidate_outcomes": sum(len(r["labels"]) for r in rows),
        "decision": expected,
        "gates": gates,
        "neural_execution": False,
    }
    if Path(output).exists():
        raise FileExistsError("Independent receipt already exists")
    save(Path(output), receipt)
    return receipt


if __name__ == "__main__":
    main()
