"""Evaluation preparation only. No optimizer, training loop, or protected dataset access."""
from dataclasses import asdict, dataclass
import hashlib
import json
from .core import STATUS


@dataclass(frozen=True)
class Protocol:
    hypothesis: str = 'Concept guidance improves held-out transition accuracy at matched exposure.'
    dataset_ids: tuple[str, ...] = ()
    seeds: tuple[int, ...] = (17, 29, 43)
    training_examples: tuple[int, ...] = (32, 128, 512)
    optimizer_steps: int = 2000
    batch_size: int = 32
    baselines: tuple[str, ...] = ('deterministic-rule', 'neural-no-concepts',
        'neural-concept-guided', 'concept-shuffled', 'concept-ablated',
        'R5.4-if-task-comparable', 'V5-shared-graph-if-task-comparable')
    splits: tuple[str, ...] = ('train', 'held-out-values', 'unseen-compositions', 'cross-domain-transfer')
    metrics: tuple[str, ...] = ('exact-transition-accuracy', 'composition-success',
        'error-by-depth', 'examples-to-target', 'robustness', 'wall-seconds', 'peak-memory')
    statistical_test: str = 'Paired bootstrap 95% CI over frozen test identities; Holm correction across comparisons'
    approval_required: bool = True

    def freeze(self) -> dict:
        if not self.dataset_ids or any(len(x) != 64 or any(c not in '0123456789abcdef' for c in x) for x in self.dataset_ids):
            raise ValueError('Freeze requires actual SHA256 dataset identities')
        if self.optimizer_steps <= 0 or self.batch_size <= 0 or not self.seeds:
            raise ValueError('Invalid experiment budget')
        payload = asdict(self)
        encoded = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
        return {'protocol': payload, 'sha256': hashlib.sha256(encoded).hexdigest(),
                'status': STATUS, 'authorization': 'NOT_AUTHORIZED_BY_FREEZE'}


def evaluate_predictions(records) -> dict:
    """Summarize supplied verifier records; cannot establish neural learning provenance."""
    records = tuple(records)
    return {'count': len(records),
            'verified_fraction': sum(r.verified for r in records) / len(records) if records else None,
            'sources': sorted({r.source for r in records}),
            'scientific_status': STATUS,
            'interpretation': 'Bounded transition checks only; no generalization or neural success claim.'}


def candidate_neural_core(state_width: int, action_width: int, concept_width: int,
                          hidden_width: int = 64):
    """Construct an optional untrained shared core. Caller controls seeds; no training occurs."""
    if min(state_width, action_width, concept_width, hidden_width) < 1:
        raise ValueError('Widths must be positive')
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError('Optional PyTorch is unavailable') from exc

    class CandidateTransitionCore(nn.Module):
        scientific_status = STATUS
        trained = False

        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(state_width + action_width + concept_width, hidden_width),
                                     nn.GELU(), nn.Linear(hidden_width, state_width))

        def forward(self, state, action, concept):
            return self.net(torch.cat((state, action, concept), dim=-1))

    return CandidateTransitionCore()
