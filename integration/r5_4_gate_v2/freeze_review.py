"""Freeze review artifacts once; no gate adoption or training authorization."""
from pathlib import Path
import json
from evidence import HERE,REPO,sha,require


def freeze():
    require(not (HERE/'LOCK.json').exists(),'Review freeze cannot be overwritten')
    files=sorted([*HERE.glob('*.py'),*HERE.glob('*.json'),HERE/'REVIEW.md',
        *[p for p in (REPO/'experiments/r5_4/gate_integration_v2').rglob('*') if p.is_file()]])
    lock=dict(version='R54_V2_GATE_INTEGRATION_REVIEW-v1',adoption_approved=False,scientific_training_authorized=False,
        sha256={str(p.relative_to(REPO)):sha(p) for p in files})
    with (HERE/'LOCK.json').open('x') as stream:
        json.dump(lock,stream,indent=2,sort_keys=True)
        stream.write('\n')


if __name__=='__main__':
    freeze()
