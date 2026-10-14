"""Externally signed content custody. No private signing keys or approval writer.

Trusted public keys must be selected independently, not from a receipt. Hashes
prove content consistency; Ed25519 proves possession of the configured signer's
key. Human identity and temporal independence remain external audit obligations.
"""
from __future__ import annotations
import base64
from dataclasses import dataclass
from collections import Counter
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from ..benchmark import digest, check_identities, FAMILIES
from .adapter import SPEC

REQUIRED = ('protocol_adoption','dataset_generation_qualification','numerical_adapter_freeze',
            'fitting_authorization','sealed_final_evaluation_authorization')
BINDINGS = ('protocol_hash','source_hash','numerical_spec_hash','input_hash','label_commitment',
            'runtime_policy_hash','train_input_hash','train_demonstration_hash','historical_registry_hash','qualification_hash')

@dataclass(frozen=True)
class VerifiedAccess:
    action: str
    binding: dict
    receipts: tuple
    trusted_keys: dict


def verify_signed(receipt, purpose, binding, trusted_keys):
    if set(receipt) != {'payload','signature'}:
        raise ValueError('signed receipt schema')
    payload=receipt['payload']
    if set(payload) != {'purpose','binding','signer','decision_id','approved','sequence'}:
        raise ValueError('receipt payload schema')
    if payload['purpose']!=purpose or payload['binding']!=binding or payload['approved'] is not True:
        raise ValueError('missing purpose/content approval')
    if not isinstance(payload['decision_id'],str) or not payload['decision_id'] or type(payload['sequence']) is not int:
        raise ValueError('missing decision evidence')
    key=trusted_keys.get((purpose,payload['signer']))
    if not isinstance(key,bytes) or len(key)!=32:
        raise ValueError('untrusted external signer')
    import json
    encoded=json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    try:
        Ed25519PublicKey.from_public_bytes(key).verify(base64.b64decode(receipt['signature'],validate=True),encoded)
    except Exception as exc:
        raise ValueError('invalid independent signature') from exc
    return payload


def authorize(action,binding,receipts,trusted_keys):
    if action not in ('fit','evaluate') or set(binding)!=set(BINDINGS):
        raise ValueError('action/binding schema')
    if binding['numerical_spec_hash']!=digest(SPEC):
        raise ValueError('numerical spec not bound')
    if any(not isinstance(v,str) or len(v)!=64 or any(c not in '0123456789abcdef' for c in v) for v in binding.values()):
        raise ValueError('invalid commitment')
    purposes=(*REQUIRED[:4],'independent_eligibility') if action=='fit' else (*REQUIRED,'independent_eligibility') if action=='evaluate' else REQUIRED[:2]
    for purpose in purposes:
        if purpose not in receipts:
            raise ValueError('missing separate human decision: '+purpose)
        verify_signed(receipts[purpose],purpose,binding,trusted_keys)
    ids=[receipts[p]['payload']['decision_id'] for p in purposes]
    if len(set(ids))!=len(ids):
        raise ValueError('separate decisions cannot reuse decision id')
    return VerifiedAccess(action,dict(binding),tuple((p,receipts[p]) for p in purposes),dict(trusted_keys))


def require_access(access,action,binding):
    if not isinstance(access,VerifiedAccess) or access.action!=action or access.binding!=binding:
        raise ValueError('no external content-bound authorization')
    # Recheck every signature at the protected boundary; frozen dataclass alone
    # cannot make nested mutable objects trustworthy.
    authorize(action,binding,dict(access.receipts),access.trusted_keys)


def observable_report(tasks,historical_ids):
    identity=check_identities(tasks,historical_ids)
    counts=Counter(t.split for t in tasks)
    if counts!={'train':768,'validation':96,'ood':216}:
        raise ValueError('frozen dataset counts')
    families=Counter(t.family for t in tasks if t.split=='ood')
    if families!=dict.fromkeys(FAMILIES,36):
        raise ValueError('frozen family count')
    # Deliberately only PUBLIC summaries; no teacher, intermediate states, answer
    # computation or local reference labels. Semantic context claims must come
    # from the independent custodian, because public identity disjointness alone
    # does not prove composition novelty or familiar constituent support.
    support={}
    for split in ('train','validation','ood'):
        members=[t for t in tasks if t.split==split]
        support[split]={'operation_counts':dict(Counter(op for t in members for op,_ in t.steps)),
                       'sequence_lengths':dict(Counter(len(t.steps) for t in members)),
                       'operand_digit_lengths':dict(Counter(len(str(n)) for t in members for n in [t.start,*[n for _,n in t.steps]]))}
    return {'identities':identity,'public_support':support,'input_hash':digest([t.public() for t in tasks]),
            'semantic_context_status':'EXTERNAL_CERTIFICATE_REQUIRED'}


def qualify(tasks,historical_ids,certificate,binding,trusted_keys):
    report=observable_report(tasks,historical_ids)
    if report['input_hash']!=binding['input_hash'] or digest(sorted(historical_ids))!=binding['historical_registry_hash']:
        raise ValueError('input/historical registry substitution')
    required={'composition_novelty','constituent_support','cross_split_leakage','historical_alias_audit','distribution_analysis','one_draw_provenance'}
    if set(certificate)!= {'claims','receipt'} or set(certificate['claims'])!=required:
        raise ValueError('complete independent semantic certificate required')
    # Claims are hashes of independent evidence, not author-issued booleans.
    for value in certificate['claims'].values():
        if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('qualification evidence hash')
    qualification={'report':report,'claims':certificate['claims']}
    if digest(qualification)!=binding['qualification_hash']:
        raise ValueError('qualification evidence not bound')
    verify_signed(certificate['receipt'],'independent_eligibility',binding,trusted_keys)
    return qualification


@dataclass(frozen=True)
class GenerationAccess:
    request: dict
    receipts: tuple
    trusted_keys: dict


def authorize_generation(request,receipts,trusted_keys):
    """Pre-draw request has no circular input/label/qualification commitment."""
    expected={'protocol_hash','source_hash','numerical_spec_hash','draw_seed','counts','families','draw_id'}
    if set(request)!=expected or request['numerical_spec_hash']!=digest(SPEC) or type(request['draw_seed']) is not int or request['draw_seed']!=520054:
        raise ValueError('frozen one-draw request')
    if request['counts']!={'train':768,'validation':96,'ood_per_family':36} or request['families']!=list(FAMILIES) or not isinstance(request['draw_id'],str) or not request['draw_id']:
        raise ValueError('frozen draw workload')
    for key in ('protocol_hash','source_hash','numerical_spec_hash'):
        value=request[key]
        if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('draw immutable hash')
    for purpose in ('protocol_adoption','dataset_generation_authorization','numerical_adapter_freeze'):
        if purpose not in receipts: raise ValueError('missing pre-draw decision')
        verify_signed(receipts[purpose],purpose,request,trusted_keys)
    return GenerationAccess(dict(request),tuple(receipts.items()),dict(trusted_keys))


def generate_once(access,marker:Path):
    """Explicit pre-draw authorization; external postdraw qualification is later.

    Failure consumes the marker and draw_id: no redraw/filter path exists.
    """
    if not isinstance(access,GenerationAccess): raise ValueError('no generation access')
    authorize_generation(access.request,dict(access.receipts),access.trusted_keys)
    with marker.open('x') as stream:
        stream.write(digest(access.request)); stream.flush()
        import os
        os.fsync(stream.fileno())
    from ..benchmark import make_public_suite
    tasks=make_public_suite(seed=520054,train_n=768,validation_n=96,family_n=36)
    return {'tasks':tasks,'input_hash':digest([t.public() for t in tasks]),
            'draw_request_hash':digest(access.request),'status':'DATA_UNQUALIFIED'}
