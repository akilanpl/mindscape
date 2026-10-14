"""Stage B candidate language encoder. Untrained; never used as a product fallback.

No pretrained parameters, external tokenizer, training run, qualification claim or
held-out vocabulary exposure is introduced by this module.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Iterable

INTENTS = ('add', 'borrow_digit', 'rename', 'force', 'tool', 'unsupported')
ARGUMENT_SLOTS = ('left', 'right', 'mass', 'force_x', 'force_y', 'velocity_x',
                  'velocity_y', 'dt', 'old_name', 'requested_name')
STATUS = 'UNTRAINED — LEARNED LANGUAGE INTERPRETATION UNAVAILABLE'
_TOKEN = re.compile(r'[A-Za-z_]+|\d|[^\w\s]')


def tokenize(text: str) -> tuple[str, ...]:
    if not isinstance(text, str) or len(text) > 16000:
        raise ValueError('Bounded text required')
    return tuple(_TOKEN.findall(text.lower()))


@dataclass(frozen=True)
class FrozenVocabulary:
    tokens: tuple[str, ...]
    training_sha256: str
    max_length: int = 128

    @classmethod
    def from_training_forms(cls, forms: Iterable[str], *, max_length=128):
        """Call with training forms only; never concatenate validation/test forms."""
        forms = tuple(forms)
        if not forms or len(forms) > 100000 or not 8 <= max_length <= 512:
            raise ValueError('Require bounded nonempty training forms and length 8..512')
        if any(not isinstance(s, str) or len(s) > 16000 for s in forms):
            raise ValueError('Invalid training form')
        words = sorted({token for form in forms for token in tokenize(form)})
        if len(words) > 20000: raise ValueError('Vocabulary budget exceeded')
        digest = hashlib.sha256(json.dumps(forms, ensure_ascii=True, separators=(',', ':')).encode()).hexdigest()
        return cls(('<pad>', '<unk>', '<bos>') + tuple(words), digest, max_length)

    def encode(self, text):
        mapping = {t: i for i, t in enumerate(self.tokens)}
        pieces = tokenize(text)
        # Truncation can discard argument text; product callers must abstain on it.
        values = [2] + [mapping.get(t, 1) for t in pieces[:self.max_length-1]]
        mask = [True] * len(values)
        padding = self.max_length-len(values)
        return {'ids': values + [0]*padding, 'mask': mask + [False]*padding,
                'truncated': len(pieces) > self.max_length-1,
                'unknown_tokens': sum(t not in mapping for t in pieces)}

    def manifest(self):
        return {'tokens': list(self.tokens), 'training_sha256': self.training_sha256,
                'max_length': self.max_length,
                'vocabulary_sha256': hashlib.sha256(json.dumps(self.tokens).encode()).hexdigest(),
                'source': 'training forms only; no pretrained tokenizer', 'status': STATUS}


def validate_split(training_forms, heldout_forms):
    """Check surface disjointness only; this does not certify semantic isolation."""
    train = tuple(training_forms); heldout = tuple(heldout_forms)
    normalize = lambda s: ' '.join(tokenize(s))
    if not train or not heldout: raise ValueError('Both splits required')
    overlap = set(map(normalize, train)) & set(map(normalize, heldout))
    if overlap: raise ValueError('Training and held-out paraphrase forms overlap')
    return {'surface_disjoint': True, 'training_count': len(train), 'heldout_count': len(heldout),
            'qualification': 'NOT_QUALIFIED',
            'limitation': 'Surface disjointness does not establish unseen operands, templates or semantic generalization.'}


def build_language_encoder(vocabulary: FrozenVocabulary, embedding_size=32, hidden_size=48):
    """Construct a random original torch encoder. Does not train or enable inference.

    Intent logits are B×6. Start/end pointer logits are B×10×T. Slot labels and
    protected requested rename targets are supplied separately by an authorized
    future training protocol; target_name is never an encoder argument head.
    """
    if not isinstance(vocabulary, FrozenVocabulary): raise TypeError('Frozen vocabulary required')
    if not 8 <= embedding_size <= 128 or not 8 <= hidden_size <= 256:
        raise ValueError('Encoder dimension budget exceeded')
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError('Optional torch dependency unavailable; no learned interpretation') from exc

    class CompactLanguageEncoder(nn.Module):
        scientific_status = STATUS
        qualified = False
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(len(vocabulary.tokens), embedding_size, padding_idx=0)
            self.encoder = nn.GRU(embedding_size, hidden_size, batch_first=True, bidirectional=True)
            self.intent_head = nn.Linear(2*hidden_size, len(INTENTS))
            self.start_head = nn.Linear(2*hidden_size, len(ARGUMENT_SLOTS))
            self.end_head = nn.Linear(2*hidden_size, len(ARGUMENT_SLOTS))
            self.vocabulary_manifest = vocabulary.manifest()

        def forward(self, token_ids, mask):
            if token_ids.ndim != 2 or token_ids.shape != mask.shape or token_ids.shape[1] > vocabulary.max_length or token_ids.shape[1] < 1:
                raise ValueError('Expected bounded B×T token IDs and mask')
            if token_ids.dtype != torch.long or mask.dtype != torch.bool:
                raise ValueError('Token IDs must be int64 and mask boolean')
            if not bool(mask.any(dim=1).all()): raise ValueError('At least one token per example required')
            if bool(((token_ids < 0) | (token_ids >= len(vocabulary.tokens))).any()):
                raise ValueError('Token ID outside frozen vocabulary')
            # Packed recurrence prevents pad length from changing contextual features.
            lengths = mask.sum(dim=1)
            expected = torch.arange(token_ids.shape[1], device=mask.device)[None, :] < lengths[:, None]
            if not torch.equal(mask, expected): raise ValueError('Mask must be a contiguous prefix')
            packed = nn.utils.rnn.pack_padded_sequence(self.embedding(token_ids), lengths.cpu(), batch_first=True, enforce_sorted=False)
            encoded, _ = self.encoder(packed)
            contextual, _ = nn.utils.rnn.pad_packed_sequence(encoded, batch_first=True, total_length=token_ids.shape[1])
            pooled = (contextual*mask.unsqueeze(-1)).sum(dim=1)/lengths.unsqueeze(-1)
            def pointers(head):
                return head(contextual).transpose(1,2).masked_fill(~mask[:,None,:], float('-inf'))
            return {'intent_logits': self.intent_head(pooled), 'argument_start_logits': pointers(self.start_head),
                    'argument_end_logits': pointers(self.end_head)}

        def interpret(self, *args, **kwargs):
            raise RuntimeError('Learned interpretation unavailable: training and qualification have not occurred')

    return CompactLanguageEncoder()
