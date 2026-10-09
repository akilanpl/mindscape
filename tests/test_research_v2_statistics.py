import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/research_v2'))
from analyze_mechanisms import exact_p, holm


def test_exact_paired_test():
    assert exact_p(0, 0) == 1
    assert exact_p(4, 0) == .125
    assert exact_p(0, 4) == .125
    assert exact_p(2, 2) == 1


def test_holm_family_monotone_and_order():
    assert holm([.04, .001, .03]) == [.06, .003, .06]
    assert holm([1., 1.]) == [1., 1.]
