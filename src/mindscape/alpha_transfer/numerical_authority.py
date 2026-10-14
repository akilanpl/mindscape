"""Purpose-specific Ed25519 verifier; pinned keys supplied out of band.

No signer, private keys or approval issuer. Key ownership/human identity must be
independently audited. Every protected boundary re-verifies content signatures.
"""
import base64
import json
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from .readiness import digest


def verify_signed(receipt,purpose,binding,keys):
    if set(receipt)!={'payload','signature'}: raise ValueError('Signed receipt schema')
    p=receipt['payload']
    if set(p)!={'purpose','binding','signer','decision_id','sequence','decision'} or p['purpose']!=purpose or p['binding']!=binding or p['decision']!='AUTHORIZE' or type(p['sequence']) is not int or not isinstance(p['decision_id'],str) or not p['decision_id']: raise ValueError('Purpose/content decision mismatch')
    key=keys.get((purpose,p['signer']))
    if not isinstance(key,bytes) or len(key)!=32: raise ValueError('Externally pinned signer missing')
    try: Ed25519PublicKey.from_public_bytes(key).verify(base64.b64decode(receipt['signature'],validate=True),json.dumps(p,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
    except Exception as exc: raise ValueError('Invalid external signature') from exc
    return p


class SignedAuthority:
    def __init__(self,keys,receipts):
        self.keys=dict(keys); self.receipts=receipts
    def binding(self,spec):
        return dict(proposal=spec['proposal_file_sha256'],bindings=spec['bindings'],candidate=spec['candidate_sha256'],training_targets=digest(spec['training_target_sha256']),permutation=digest(spec['permutation']),schedules=digest(spec['schedules']),qualification=spec['qualification'])
    def verify(self,spec,manifest,candidate):
        if spec['bindings']['data']!=digest(manifest) or spec['candidate_sha256']!=digest(candidate): raise ValueError('Candidate/data changed')
        binding=self.binding(spec); ids=[]
        for purpose in ('candidate_freeze','data_qualification','training'):
            ids.append(verify_signed(self.receipts[purpose],purpose,binding,self.keys)['decision_id'])
        if len(set(ids))!=3: raise ValueError('Distinct stage decisions required')
        return True
    def verify_stage(self,purpose,spec,seals):
        if purpose not in ('prediction','final_evaluation'): raise ValueError('Unknown protected stage')
        binding=self.binding(spec)
        if purpose=='final_evaluation': binding={**binding,'nine_seals_sha256':digest(seals)}
        verify_signed(self.receipts[purpose],purpose,binding,self.keys)
        return True


def verify_verdict(report,spec,seals,receipt,keys):
    """Gate computation alone cannot issue a scientific verdict."""
    binding=dict(report_sha256=digest(report),source_sha256=spec['bindings']['source'],nine_seals_sha256=digest(seals),spec_sha256=digest(spec))
    verify_signed(receipt,'independent_metric_audit',binding,keys)
    return dict(status='GO_TO_CONFIRMATION' if report['status']=='GATES_PASS_PENDING_INDEPENDENT_AUDIT' else 'NO_GO',independent_audit_sha256=digest(receipt))
