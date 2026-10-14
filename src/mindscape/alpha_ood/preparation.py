"""Fail-closed preparation receipts. This namespace cannot launch scientific fits."""
from .benchmark import digest

REQUIRED = ("human_protocol_adoption", "human_separate_training_authorization",
            "independent_source_lock_review", "historical_identity_registry",
            "custodian_context_eligibility_certificate",
            "frozen_learned_model_adapter_and_initialization_receipts",
            "dataset_input_and_label_commitments",
            "independent_prediction_seal_before_label_release")


def readiness(protocol: dict, receipts: dict) -> dict:
    missing = [key for key in REQUIRED if not isinstance(receipts.get(key), str) or
               len(receipts[key]) != 64 or any(c not in "0123456789abcdef" for c in receipts[key])]
    return {"scientific_status": "UNRUN", "data_status": "DATA_UNQUALIFIED" if missing else "REVIEW_REQUIRED",
            "execution_permitted": False, "missing_receipts": missing,
            "protocol_hash": digest(protocol),
            "reason": "Receipt hashes are references, not authorization authentication; independent human review required."}


def fit_plan(protocol: dict) -> tuple[dict, ...]:
    """Exact proposed workload, not executable authorization or optimizer updates."""
    if protocol["learned_fits"] != 6 or protocol["updates_per_fit"] != 800:
        raise ValueError("unexpected frozen workload")
    return tuple({"condition": condition, "seed": seed, "updates": 800,
                  "batch_size": 32, "checkpoint": "final_only", "status": "UNRUN"}
                 for condition in ("recurrent", "feedforward") for seed in protocol["fit_seeds"])
