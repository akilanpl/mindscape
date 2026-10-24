import subprocess
import sys

import pytest

from research.r6.datasets import coding


def test_pilot_reproducibility_bounds_and_metadata():
    for split, count in [('train', 512), ('dev', 64), ('final_iid', 96), ('final_ood', 96)]:
        rows = coding.generate(split, 71, count)
        assert rows == coding.generate(split, 71, count)
        assert len(rows) == len({r['record_id'] for r in rows}) == len({r['canonical_key'] for r in rows})
        for row in rows:
            assert row['domain'] == 'coding' and row['source'] == 'generated'
            assert row['split'] == split and row['generator_version'] == 'r6-v1'
            assert len(row['input'].encode('ascii')) <= 192
            assert row['features'] == []
            assert str(coding.interpret(row['input'])) == row['target']['answer']


def test_global_split_separation_and_structural_ood():
    by_split = {s: coding.generate(s, 91+i, 150) + coding.generate(s, 109+i, 150)
                for i, s in enumerate(coding.SPLITS)}
    for i, first in enumerate(coding.SPLITS):
        a = {r['canonical_key'] for r in by_split[first]}
        for second in coding.SPLITS[i+1:]:
            assert not a.intersection(r['canonical_key'] for r in by_split[second])
    assert not {r['template_id'] for r in by_split['train']}.intersection(
        r['template_id'] for r in by_split['final_ood'])


def test_renaming_whitespace_and_input_dependence():
    for split in coding.SPLITS:
        for row in coding.generate(split, 17, 28):
            probes = coding.perturbations(row)
            for name in ('renamed', 'reformatted', 'control_flow_equivalent'):
                assert probes[name]['target'] == row['target']
            for name in ('renamed', 'reformatted'):
                assert probes[name]['canonical_key'] == row['canonical_key']
            assert probes['literal_intervention']['target'] != row['target']
            assert probes['control_flow_equivalent']['canonical_key'] != row['canonical_key']


def test_reference_cpython_agrees_on_trusted_generated_templates():
    # Only programs constructed by our closed generator are sent to CPython.
    # -I isolates imports/environment; strict timeout bounds independent checks.
    # No caller-provided source or downloaded program is executed here.
    for split in ('train', 'final_ood'):
        for row in coding.generate(split, 18, 12):
            binding = row['structured_input']['output_binding']
            result = subprocess.run([sys.executable, '-I', '-c', row['input'] + '\nprint('+binding+')'],
                                    capture_output=True, text=True, timeout=2, check=True)
            assert result.stdout.strip() == row['target']['answer']


@pytest.mark.parametrize('program', [
    'import os\nx=1', 'x=__import__("os")', 'x=(1).__class__',
    'x=[i for i in range(5)]', 'while 1:\n x=1', 'x=2**1000000',
    'for x in range(1000000):\n y=1'])
def test_restricted_interpreter_rejects_unsafe_or_unbounded(program):
    with pytest.raises((ValueError, KeyError, TypeError)):
        coding.interpret(program)


@pytest.mark.parametrize('split,count', [('test', 1), ('train', -1), ('train', 1001), ('train', 1.2)])
def test_invalid_requests(split, count):
    with pytest.raises(ValueError):
        coding.generate(split, 17, count)


def test_empty_request():
    assert coding.generate('train', 17, 0) == []


def test_acceptance_statistics_are_actual_generation_counts():
    rows = coding.generate('train', 71, 512)
    stats = coding.generation_statistics('train', 71, 512)
    assert stats['accepted'] == len(rows)
    assert stats['attempted'] == stats['accepted'] + stats['rejected_canonical_duplicate']
    assert stats['unique_templates'] == 4
    assert stats['input_ascii_bytes']['total'] == sum(len(r['input']) for r in rows)
