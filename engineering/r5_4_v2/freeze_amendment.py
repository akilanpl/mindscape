"""Create the separate engineering freeze once; never execute qualification."""
from pathlib import Path
from common import REPO,HERE,sha,load,write_new


def create():
    destination=HERE/'LOCK.json'
    if destination.exists():
        raise ValueError('Existing engineering freeze cannot be overwritten')
    evidence=REPO/'experiments/r5_4/engineering_v2'
    files=sorted([*HERE.glob('*.py'),HERE/'plan.json',HERE/'AMENDMENT.md',
        evidence/'diagnostic-independent-audit.json',evidence/'nontraining-tests.txt',*list((evidence/'evidence').glob('*')),
        REPO/'experiments/r5_4/engineering.json',REPO/'diagnostics/r5_4_mps_v1/run.py',
        REPO/'diagnostics/r5_4_mps_v1/schedule.json',REPO/'diagnostics/r5_4_mps_v1/user-reported-native-engineering.json'])
    original=REPO/'experiments/r5_4/frozen/LOCK.json'
    write_new(destination,dict(version='R5.4-ENGINEERING-QUALIFICATION-v2',status='FROZEN_FOR_REVIEW_NOT_EXECUTED',
        v1_lock_sha256=sha(original),native_diagnostic_receipt_sha256=load(evidence/'diagnostic-independent-audit.json')['receipt_sha256'],
        sha256={str(p.relative_to(REPO)):sha(p) for p in files}))


if __name__=='__main__':
    create()
