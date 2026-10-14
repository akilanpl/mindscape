"""Fail-closed reusable evidence checks; receipts are consistency evidence, not identity attestation."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist


class AuditError(ValueError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_inventory(root, inventory):
    """Verify only supplied inventory, rejecting traversal and symlink ancestors."""
    root = Path(root).absolute()
    if not inventory:
        raise AuditError("empty inventory")
    for relative, expected in inventory.items():
        rel = Path(relative)
        if rel.is_absolute() or not rel.parts or any(p in ("..", ".") for p in rel.parts):
            raise AuditError("unsafe inventory path")
        path = root / rel
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise AuditError("symlink inventory path")
        if not path.is_file() or file_hash(path) != expected:
            raise AuditError(f"source/evidence mismatch: {relative}")
    return {"status": "VERIFIED", "files": len(inventory), "inventory_sha256": digest(inventory)}


def verify_disjoint(splits, historical=()):
    seen = set(historical)
    if len(seen) != len(historical):
        raise AuditError("duplicate historical identities")
    counts = {}
    for name, identities in splits.items():
        values = list(identities)
        if not values or any(not isinstance(x, str) or not x for x in values):
            raise AuditError("empty/invalid identities")
        if len(set(values)) != len(values) or seen.intersection(values):
            raise AuditError(f"identity leakage: {name}")
        seen.update(values)
        counts[name] = len(values)
    if not counts:
        raise AuditError("no splits")
    return counts


def verify_fit_matching(receipts, seeds, conditions, updates, target_presentations):
    """Compare per-seed init, target sequence and optimizer configuration across conditions."""
    expected = {(s, c) for s in seeds for c in conditions}
    observed = {}
    for receipt in receipts:
        key = (receipt["seed"], receipt["condition"])
        if key in observed or key not in expected:
            raise AuditError("duplicate/unexpected fit")
        for field in ("initial_state_sha256", "target_order_sha256", "optimizer_config_sha256"):
            value = receipt.get(field)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise AuditError(f"missing/invalid {field}")
        if type(receipt.get("updates")) is not int or receipt["updates"] != updates:
            raise AuditError("optimizer budget mismatch")
        if type(receipt.get("target_presentations")) is not int or receipt["target_presentations"] != target_presentations:
            raise AuditError("target exposure mismatch")
        if receipt.get("checkpoint_rule") != "final":
            raise AuditError("checkpoint selection mismatch")
        observed[key] = receipt
    if set(observed) != expected:
        raise AuditError("missing fit; unsuccessful seeds must be preserved")
    for seed in seeds:
        group = [observed[(seed, c)] for c in conditions]
        for field in ("initial_state_sha256", "target_order_sha256", "optimizer_config_sha256"):
            if len({r[field] for r in group}) != 1:
                raise AuditError(f"unmatched {field}")
    return {"status": "VERIFIED", "fits": len(observed)}


def verify_event_chain(events, prediction_file):
    """Verify tamper-evident event order and sealed bytes before any reference access.

    A local chain is not a trusted clock or proof that labels were never accessed elsewhere.
    Custody and process-isolation evidence must be reviewed separately.
    """
    previous = "0" * 64
    sealed = False
    scored = False
    for index, event in enumerate(events):
        if event.get("sequence") != index or event.get("previous_sha256") != previous:
            raise AuditError("broken event sequence")
        payload = {k: v for k, v in event.items() if k != "sha256"}
        if event.get("sha256") != digest(payload):
            raise AuditError("event hash mismatch")
        previous = event["sha256"]
        kind = event.get("event")
        if kind == "PREDICTIONS_SEALED":
            if sealed or scored or event.get("prediction_sha256") != file_hash(prediction_file):
                raise AuditError("invalid prediction seal")
            sealed = True
        if kind in ("REFERENCES_OPENED", "SCORED"):
            if not sealed:
                raise AuditError("references/scoring preceded prediction seal")
            scored |= kind == "SCORED"
        if sealed and kind in ("TRAINING", "PREDICTING", "CHECKPOINT_SELECTED"):
            raise AuditError("model/prediction mutation after seal")
    if not sealed or not scored:
        raise AuditError("incomplete seal/scoring evidence")
    return {"status": "VERIFIED", "events": len(events), "chain_sha256": previous}


def exact_metrics(predictions, references):
    """Replay whole-answer and intermediate correctness; IDs are unique tasks, never seed replicas."""
    if not references or set(predictions) != set(references):
        raise AuditError("missing/extra task predictions")
    whole = intermediate = denominator = 0
    for identity, reference in references.items():
        prediction = predictions[identity]
        truth = reference["states"]
        observed = prediction["states"]
        if not isinstance(truth, list) or not truth or not isinstance(observed, list):
            raise AuditError("invalid trace")
        if any(type(x) is not int for x in truth + observed):
            raise AuditError("invalid/noninteger state")
        if type(prediction["answer"]) is not int or type(reference["answer"]) is not int:
            raise AuditError("invalid answer")
        whole += prediction["answer"] == reference["answer"]
        intermediate += sum(a == b for a, b in zip(observed, truth))
        denominator += len(truth)
    return {"unique_tasks": len(references), "whole_answer_correct": whole,
            "whole_answer_exactness": whole / len(references),
            "intermediate_correct": intermediate, "intermediate_denominator": denominator,
            "intermediate_accuracy": intermediate / denominator}


def wilson(successes, total, confidence=0.95):
    if type(total) is not int or type(successes) is not int or total <= 0 or not 0 <= successes <= total:
        raise AuditError("invalid independent denominator")
    if not 0 < confidence < 1:
        raise AuditError("invalid confidence")
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    p = successes / total
    scale = 1 + z*z / total
    center = (p + z*z / (2*total)) / scale
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / scale
    return [max(0.0, center-half), min(1.0, center+half)]


def scientific_status(*, executed, complete, independent, gates, mandatory_gates):
    if not executed:
        return "UNRUN"
    if not complete:
        return "FAILED_CAMPAIGN"
    if not independent:
        return "UNVERIFIED"
    if not mandatory_gates or set(gates) != set(mandatory_gates) or any(type(v) is not bool for v in gates.values()):
        raise AuditError("mandatory frozen gates missing or malformed")
    return "VERIFIED_PASS" if all(gates.values()) else "VERIFIED_NO_GO"
