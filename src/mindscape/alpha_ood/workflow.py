"""Protocol-bound, serialized callback workflow; no built-in numerical trainer.

Protected callbacks are unreachable without independently validated, content-bound
review receipts. The numerical loss/exposure implementation remains review-required.
"""
from collections.abc import Callable

from .benchmark import digest
from .preparation import REQUIRED, fit_plan


class BoundaryError(ValueError):
    pass


def execute_prepared_campaign(*, protocol: dict, source_hash: str, input_hash: str,
                              label_commitment: str, receipts: dict,
                              verify_receipt: Callable[[str, dict], bool],
                              fit: Callable[[dict], object],
                              predict: Callable[[object, dict], list[dict]],
                              persist_seal: Callable[[dict], None]) -> dict:
    """Extension point, not permission to run. Must be independently reviewed.

    Every signed receipt binds the same protocol/source/input/label hashes. A human
    adopter must also attest a frozen numerical adapter; this package lacks a final
    training loss/teacher-forcing specification and does not provide such an adapter.
    Prediction callbacks receive no released labels. Labels cannot be scored here.
    Failed fits abort and are retained in the returned failure receipt; no retries.
    """
    binding = {"protocol_hash": digest(protocol), "source_hash": source_hash,
               "input_hash": input_hash, "label_commitment": label_commitment}
    for value in binding.values():
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise BoundaryError("invalid immutable content hash")
    for key in REQUIRED:
        receipt = receipts.get(key)
        if not isinstance(receipt, dict) or receipt.get("binding") != binding:
            raise BoundaryError(f"missing/mismatched receipt: {key}")
        if receipt.get("approved") is not True or not verify_receipt(key, receipt):
            raise BoundaryError(f"unverified independent approval: {key}")
    plan = fit_plan(protocol)
    if len(plan) != 6 or tuple(protocol["fit_seeds"]) != (11, 23, 47):
        raise BoundaryError("frozen fit design mismatch")
    if protocol["batch_size"] != 32 or protocol["checkpoint"] != "final_only":
        raise BoundaryError("frozen exposure/checkpoint mismatch")
    if protocol["optimizer"] != {"name": "Adam", "learning_rate": .003}:
        raise BoundaryError("frozen optimizer mismatch")
    predictions = []
    completed = []
    try:
        for specification in plan:  # Native jobs are intentionally serialized.
            checkpoint = fit(specification)
            rows = predict(checkpoint, specification)
            predictions.extend(rows)
            completed.append(specification)
    except Exception as exc:  # noqa: BLE001 - preserve any callback campaign failure, never retry
        failure = {"scientific_status": "FAILED_CAMPAIGN", "binding": binding,
                   "completed": completed, "predictions": predictions,
                   "failure": {"type": type(exc).__name__, "message": str(exc)}}
        persist_seal(failure)
        return failure
    sealed = {"scientific_status": "UNVERIFIED", "binding": binding,
              "completed": completed, "predictions": predictions,
              "prediction_hash": digest(predictions), "labels_released": False}
    persist_seal(sealed)
    return sealed
