"""Kaggle runtime-isolation guards, anchored to trusted controller evidence.

The account owner remains the trusted controller. These guards protect a frozen
trainer execution from evaluation inputs; they do not constrain future owner edits.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

CREDENTIAL_NAMES = ('KAGGLE_KEY', 'KAGGLE_API_TOKEN', 'KAGGLE_USERNAME',
                    'AWS_SECRET_ACCESS_KEY', 'GITHUB_TOKEN', 'GH_TOKEN', 'HF_TOKEN',
                    'GOOGLE_APPLICATION_CREDENTIALS')


def evidence_digest(evidence):
    return hashlib.sha256(json.dumps(evidence, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def credential_guard(environ=None, home=None):
    environ = os.environ if environ is None else environ
    if any(environ.get(key) for key in CREDENTIAL_NAMES):
        raise PermissionError('Known cloud/training credential environment variable present')
    home = Path.home() if home is None else Path(home)
    for relative in ('.kaggle/kaggle.json', '.kaggle/access_token', '.kaggle/oauth_credentials.json', '.config/kaggle/access_token',
                     '.config/gh/hosts.yml', '.aws/credentials'):
        if (home / relative).exists():
            raise PermissionError('Known credential file present')


def readonly_input(path, input_root='/kaggle/input'):
    path = Path(path).resolve(); root = Path(input_root).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise PermissionError('Finalized inputs must be files under Kaggle input mount')
    # Opening O_WRONLY without O_TRUNC/O_CREAT changes no bytes. It must fail on
    # the provider's read-only input mount, regardless of Unix permission bits.
    try:
        descriptor = os.open(path, os.O_WRONLY)
    except OSError as exc:
        import errno
        if exc.errno not in (errno.EROFS, errno.EACCES, errno.EPERM):
            raise PermissionError('Could not establish read-only checkpoint access') from exc
    else:
        os.close(descriptor)
        raise PermissionError('Checkpoint/input permits writes; mount is not protected')
    return str(path)


def validate_controller_evidence(evidence):
    if evidence.get('provider') != 'kaggle' or evidence.get('protection_scope') != 'frozen-runtime-isolation':
        raise PermissionError('Kaggle runtime evidence required')
    trainer = evidence.get('trainer', {}); evaluator = evidence.get('evaluator', {})
    if not trainer.get('kernel_id') or not evaluator.get('kernel_id') or trainer['kernel_id'] == evaluator['kernel_id']:
        raise PermissionError('Distinct trainer/evaluator kernel executions required')
    for role in (trainer, evaluator):
        if not role.get('version') or not role.get('source_sha256') or not role.get('remote_metadata_sha256'):
            raise PermissionError('Pinned remotely retrieved source, metadata and version required')
        metadata = role.get('metadata', {})
        if metadata.get('id') != role['kernel_id'] or metadata.get('is_private') is not True or metadata.get('enable_internet') is not False:
            raise PermissionError('Both frozen kernels must be verified private and offline')
        if role.get('credential_access_code_absent') is not True:
            raise PermissionError('Frozen source must not request credentials or user secrets')
    if evaluator.get('source_identity_scope') != 'evaluator-core-file' or not evaluator.get('source_path'):
        raise PermissionError('Evaluator source identity must pin core file, excluding launcher')
    if not evaluator.get('core_source_files') or evaluator['core_source_files'].get(evaluator['source_path']) != evaluator['source_sha256']:
        raise PermissionError('Evaluator core source inventory/entrypoint mismatch')
    if evaluator['metadata'].get('enable_gpu') is not False or evaluator['metadata'].get('enable_tpu') is not False:
        raise PermissionError('Independent evaluator must use CPU execution')
    if trainer.get('payload_evaluation_records') != 0 or trainer.get('evaluation_inputs_attached') is not False:
        raise PermissionError('Trainer package or mounted sources expose evaluation records')
    if trainer.get('control_plane_source_verified') is not True or evaluator.get('control_plane_source_verified') is not True:
        raise PermissionError('Controller must verify CLI-pulled provider source identities')
    if evidence.get('checkpoint_export_hash_verified') is not True or not evidence.get('checkpoint_input_version'):
        raise PermissionError('Exported immutable checkpoint version/hash required')
    if evidence.get('controller_trusted') is not True:
        raise PermissionError('Trusted account-owner controller required')
    return True


def kaggle_verifier(expected_evidence_sha256, readonly_inputs, *, input_root='/kaggle/input',
                    environ=None, home=None, core_root=None):
    """Build a verifier with an evidence digest pinned by the trusted controller.

    The controller must obtain metadata/source from authenticated Kaggle pulls,
    verify source/package inventories, and embed this digest in evaluator source
    before submission. A self-authored evidence file without that provider check
    does not establish isolation. No owner-account separation is claimed.
    """
    def verify(evidence):
        if evidence_digest(evidence) != expected_evidence_sha256:
            raise PermissionError('Controller evidence digest mismatch')
        validate_controller_evidence(evidence)
        if core_root is None:
            raise PermissionError('Extracted evaluator core source root required')
        root = Path(core_root).resolve()
        for relative, expected in evidence['evaluator']['core_source_files'].items():
            source = (root / relative).resolve()
            if not source.is_relative_to(root) or not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                raise PermissionError('Evaluator core source hash mismatch')
        current = os.environ if environ is None else environ
        if current.get('KAGGLE_KERNEL_RUN_TYPE') != 'Batch':
            raise PermissionError('Separate Kaggle batch execution required')
        credential_guard(current, home)
        if not readonly_inputs:
            raise PermissionError('Immutable checkpoint and evaluation inputs required')
        protected = [readonly_input(path, input_root) for path in readonly_inputs]
        verify.runtime_audit = {
            'provider_batch_execution': True,
            'known_credential_environment_absent': True,
            'known_credential_files_absent': True,
            'checkpoint_write_open_denied_paths': protected,
            'core_source_files_verified': len(evidence['evaluator']['core_source_files']),
            'controller_evidence_sha256': expected_evidence_sha256,
            'protection_scope': 'frozen-runtime-isolation',
            'controller_trusted': True,
            'owner_can_change_future_runs': True,
        }
        return True
    return verify
