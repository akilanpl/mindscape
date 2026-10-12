"""Fair baseline declarations and budget compatibility gates."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingBudget:
    backbone_hash: str
    training_task_ids: tuple[str, ...]
    updates: int
    supervised_labels: int
    verifier_queries: int
    tokens: int = 0

    def __post_init__(self):
        if min(self.updates, self.supervised_labels, self.verifier_queries, self.tokens) < 0:
            raise ValueError("Negative training budget")
        if len(set(self.training_task_ids)) != len(self.training_task_ids):
            raise ValueError("Duplicate training identities")


def require_fair(left: TrainingBudget, right: TrainingBudget):
    if left != right:
        raise ValueError("Unmatched backbone, information or training budget; label as confounded")


BASELINE_MATRIX = {
    "random": {"backbone": "tiny_mlp", "updates": 0, "feedback": 0, "status": "implemented"},
    "zero": {"backbone": "tiny_mlp", "updates": 0, "feedback": 0, "status": "implemented"},
    "answer_only": {"backbone": "tiny_mlp", "status": "interface_only_not_trained"},
    "structured": {"backbone": "tiny_mlp", "status": "interface_only_not_trained"},
    "episodic": {"backbone": "tiny_mlp", "status": "interface_only_not_trained"},
    "gpt_neo": {"backbone": "local_historical", "status": "optional_unrun_requires_approved_files"},
    "pythia": {"backbone": "local_historical", "status": "optional_unrun_requires_approved_files"},
}
