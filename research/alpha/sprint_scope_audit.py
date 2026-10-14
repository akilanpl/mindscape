"""Independent storage-only full-workload/source audit. No scientific module imports."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

PINS = {
    'r54': '950a17519f43fbc887e41f29ce2e24d4b6036ead',
    'ood': 'd457428540b34a96e16c31fc4ffe3d13e8cba3fa',
    'transfer': '03fc481192dca47410dd809de1303658c2f556cb',
}
LOCKS = {
    'experiments/r5_4/frozen/LOCK.json': 'a771f8ae619fa080b09f49c52ebbad6f8eb2dbb37f6b4b6748fbd86700f6a30a',
    'engineering/r5_4_v2/LOCK.json': 'ca6f9bf67ecf9fea9d3b0c8075cf3c0c0c4001f1a9049fef1cff28e5f7c585e5',
    'integration/r5_4_gate_v2/LOCK.json': '35be4554909e822458f818be5dec845c10c2cf2062582151d9dc73e244fe03c7',
}
PROTOCOLS = {
    'r54': ('experiments/r5_4/frozen/protocol.json', '8d94b1b6f495c5e4c2f4c57f31f8504ea2835d530a58779230cbc2b098090891'),
    'ood': ('configs/alpha_ood/protocol-v1.json', '70deef2ae0c667378a541d1948780fe3e78c06f421097518feebf2576fe0ea69'),
    'transfer': ('configs/r5_kernel/shared-study-proposal.json', 'a49a05def8cf588e82e0aa62fd32f988266fac3917cc06b8e3c61ca8b05adc64'),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_verified(root, relative, expected):
    rel = Path(relative)
    require(not rel.is_absolute() and '..' not in rel.parts, 'Unsafe inventory path')
    path = root / rel
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'Symlink source')
    data = path.read_bytes()
    require(sha(data) == expected, f'Frozen bytes changed: {relative}')
    return data


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args])


def audit_one(root, kind):
    supplied = Path(root).absolute()
    require(not any(p.is_symlink() for p in (supplied, *supplied.parents)), 'Symlink checkout root')
    root = supplied.resolve()
    require(not git(root, 'status', '--porcelain', '--untracked-files=all').strip(), 'Candidate worktree is dirty')
    pin = PINS[kind]
    require(subprocess.run(['git', '-C', str(root), 'merge-base', '--is-ancestor', pin, 'HEAD'],
                           capture_output=True).returncode == 0, 'Wrong source ancestry')
    changed = git(root, 'diff', '--name-status', pin, 'HEAD').decode().splitlines()
    require(all(row.startswith('A\t') for row in changed), 'Sprint must preserve every pinned file')
    # Check working bytes for every immutable ancestor file, not only committed diffs.
    entries = git(root, 'ls-tree', '-rz', pin).split(b'\0')
    entries = [entry for entry in entries if entry]
    parsed = []
    for entry in entries:
        metadata, name = entry.split(b'\t', 1)
        mode, object_type, oid = metadata.split()
        require(object_type == b'blob' and mode in (b'100644', b'100755'), 'Unexpected baseline object')
        parsed.append((oid, name.decode()))
    # One batch process avoids thousands of independent Git invocations.
    payload = subprocess.check_output(['git', '-C', str(root), 'cat-file', '--batch'],
                                     input=b''.join(oid + b'\n' for oid, _ in parsed))
    cursor = 0
    baseline = {}
    for oid, name in parsed:
        boundary = payload.index(b'\n', cursor)
        returned_oid, object_type, raw_size = payload[cursor:boundary].split()
        require(returned_oid == oid and object_type == b'blob', 'Unexpected batch source')
        size = int(raw_size)
        data = payload[boundary+1:boundary+1+size]
        require(len(data) == size and payload[boundary+1+size:boundary+2+size] == b'\n', 'Truncated source batch')
        cursor = boundary+2+size
        expected = sha(data)
        read_verified(root, name, expected)
        baseline[name] = expected
    require(cursor == len(payload), 'Extra source batch data')
    relative, expected = PROTOCOLS[kind]
    protocol = json.loads(read_verified(root, relative, expected))
    lock_receipts = []
    if kind == 'r54':
        for name, expected in LOCKS.items():
            lock = json.loads(read_verified(root, name, expected))
            for member, digest in lock['sha256'].items():
                read_verified(root, member, digest)
            lock_receipts.append({'path': name, 'sha256': expected, 'members': len(lock['sha256'])})
        require(protocol['fits'] == 6 and protocol['updates_per_fit'] == 800
                and protocol['batch_size'] == 32 and protocol['seeds'] == [11,23,47], 'R54 workload')
        workload = {'fits': 6, 'updates_per_fit': 800, 'total_updates': 4800,
                    'batch_size': 32, 'counts': protocol['counts'], 'seeds': protocol['seeds']}
        gates = protocol['gates_per_seed_per_model']
    elif kind == 'ood':
        require(protocol['counts'] == {'train':768, 'validation':96, 'ood_per_family':36}
                and len(protocol['families']) == 6 and protocol['learned_fits'] == 6
                and protocol['updates_per_fit'] == 800 and protocol['batch_size'] == 32
                and protocol['fit_seeds'] == [11,23,47], 'OOD workload')
        workload = {'fits':6,'updates_per_fit':800,'total_updates':4800,'batch_size':32,
                    'train':768,'validation':96,'ood':216,'family_counts':dict.fromkeys(protocol['families'],36),
                    'conditions':protocol['conditions'],'seeds':protocol['fit_seeds']}
        gates = protocol['gates']
    else:
        budget = protocol['budget']
        require(budget['fits'] == 9 and budget['seeds'] == [11,23,47]
                and budget['total_update_slots'] == 2160 and budget['batch_size'] == 16
                and budget['target_updates_per_fit'] == 120
                and budget['auxiliary_updates_each_domain'] == 60, 'Transfer workload')
        workload = {**budget, 'per_fit_slots':240,'architecture_parameters':6967,
                    'tasks_per_domain':protocol['data']['tasks_per_domain']}
        gates = protocol['gates']
    candidate_files = {}
    for row in changed:
        relative = row.split('\t', 1)[1]
        path = root / relative
        require(not any(p.is_symlink() for p in (path, *path.parents)), 'Symlink candidate')
        candidate_files[relative] = sha(path.read_bytes())
    return {'candidate_file_sha256':candidate_files, 'scope_status':'VERIFIED_UNCHANGED_FULL_WORKLOAD','scientific_status':'UNRUN',
            'review_pin':pin,'head':git(root,'rev-parse','HEAD').decode().strip(),
            'protocol_sha256':PROTOCOLS[kind][1],'workload':workload,'mandatory_gates':gates,
            'source_lock_checks':lock_receipts,'baseline_files_verified':len(baseline),
            'baseline_inventory_sha256':sha(json.dumps(baseline,sort_keys=True).encode()),
            'limitation':'Stored-byte/scope consistency only; not approval, execution or scientific success.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in PINS: parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({kind:audit_one(getattr(args,kind),kind) for kind in PINS},indent=2,sort_keys=True))
