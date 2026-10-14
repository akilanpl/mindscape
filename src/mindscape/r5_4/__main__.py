"""Explicit commands; default invocation performs no work."""
import argparse
import json
from pathlib import Path
from .data import ROOT, canonical, freeze_data, qualify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['generate', 'qualify', 'freeze', 'engineering', 'train', 'audit', 'preserve', 'verify-archive'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--authorize-training', action='store_true')
    parser.add_argument('--replay-mps', action='store_true')
    args = parser.parse_args()
    if args.command == 'generate':
        if (ROOT / 'LOCK.json').exists():
            raise SystemExit('Frozen data exists; regeneration prohibited')
        freeze_data()
        value = qualify()
    elif args.command == 'qualify':
        value = qualify()
    elif args.command == 'freeze':
        from .freeze import make_lock
        if (ROOT / 'LOCK.json').exists():
            raise SystemExit('Freeze exists; no automatic protocol replacement')
        value = make_lock()
    elif args.command == 'engineering':
        from .freeze import verify_lock
        from .engineering import check
        verify_lock()
        value = check(json.loads((ROOT / 'train.json').read_text()))
    elif args.command == 'train':
        if not args.output:
            parser.error('--output new campaign directory required')
        from .campaign import run
        value = run(args.output, args.authorize_training)
    elif args.command == 'preserve':
        if not args.output or not args.archive:
            parser.error('--output campaign and --archive new archive required')
        from .preservation import preserve
        value = preserve(args.output, args.archive)
    elif args.command == 'verify-archive':
        if not args.archive:
            parser.error('--archive required')
        from .preservation import verify_archive
        value = verify_archive(args.archive)
    else:
        if not args.output:
            parser.error('--output campaign directory required')
        from .audit import audit
        value = audit(args.output, args.replay_mps)
    print(canonical(value), end='')
    if value.get('status') in ('DATA_NO_GO', 'ENGINEERING_NO_GO'):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
