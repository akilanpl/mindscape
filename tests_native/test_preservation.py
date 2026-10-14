"""Native correction must preserve old study/workflow blobs without rewriting gates."""
import importlib.util
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('native_ci_preservation', ROOT/'scripts/native/ci_preservation.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_exact_historical_readme_divergence_is_preexisting():
    result = audit.readme_audit(ROOT)
    assert result['readme_git_blob_ids'] == {
        'frozen_v3': 'bac8aa0ed64a86c476d5953d8c94c88750d6214c',
        'pr27_base': '7d47768a22505341e7768da5dd1b3c9bd5558eca',
        'pr27_head': '7d47768a22505341e7768da5dd1b3c9bd5558eca'}
    assert result['preexisting_frozen_readme_divergence']


def test_all_protected_release_baseline_files_unchanged():
    result = audit.verify(ROOT)
    assert result['protected_baseline_files'] > 1200
    assert result['protected_baseline_unchanged'], result['changed_or_missing']
    assert not result['historical_checks_replaced']
    assert result['declared_historical_tags'] == 2
    assert result['changed_or_missing_tags'] == []


def test_audit_detects_frozen_source_and_lock_change(tmp_path):
    def git(*args): return subprocess.check_output(['git','-C',str(tmp_path),*args])
    git('init','-q')
    for path,content in [('configs/old-study/lock.txt','frozen'),('src/old_study.py','original'),('src/mindscape/mind1/runtime.py','mutable')]:
        file = tmp_path/path; file.parent.mkdir(parents=True,exist_ok=True); file.write_text(content)
    git('add','.')
    git('-c','user.name=Audit test','-c','user.email=audit@example.invalid','commit','-qm','baseline')
    baseline = git('rev-parse','HEAD').decode().strip()
    assert audit.verify(tmp_path,baseline)['protected_baseline_unchanged']
    (tmp_path/'src/mindscape/mind1/runtime.py').write_text('new product')
    assert audit.verify(tmp_path,baseline)['protected_baseline_unchanged']
    (tmp_path/'configs/old-study/lock.txt').write_text('weakened')
    (tmp_path/'src/old_study.py').unlink()
    result = audit.verify(tmp_path,baseline)
    assert not result['protected_baseline_unchanged']
    assert result['changed_or_missing'] == ['configs/old-study/lock.txt','src/old_study.py']


def test_failed_historical_ci_sources_precede_product_branch():
    paths = ['.github/workflows/tests.yml', '.github/workflows/v3-foundation.yml',
             'configs/research_v2/linux-cpu-lock.txt', 'scripts/research_v2/cpu_validate.py',
             'scripts/v3/preservation.py', 'tests/r5_2_recovery/test_recovery.py',
             'tests/r5_3_recovery/test_recovery.py', 'tests/r5_3_physics/test_public.py', 'pyproject.toml']
    for path in paths:
        before = audit.git(ROOT,'rev-parse',audit.PR27_BASE+':'+path)
        after = audit.git(ROOT,'rev-parse',audit.BASELINE+':'+path)
        assert before == after, path
