import pytest

from mindscape.coding.statistics import der, log_aulc, paired_delta, threshold


def test_censored_and_zero_thresholds_are_not_invented_ratios():
    assert threshold({10: 0.7, 100: 0.79}, 0.8)["right_censored"]
    zero = threshold({0: 0.9, 10: 0.95}, 0.8)
    assert zero["budget"] == 0
    assert der(zero, zero)["value"] is None
    assert der(threshold({10: 0.9}, 0.8), threshold({5: 0.9}, 0.8))["value"] == 2


def test_task_pairing_and_log_normalization():
    assert paired_delta({"a": 0, "b": 1}, {"a": 1, "b": 1})["mean"] == 0.5
    with pytest.raises(ValueError):
        paired_delta({"a": 0}, {"b": 0})
    assert log_aulc({0: 1, 10: 0.5, 100: 0.5}) == pytest.approx(0.5)


def test_perfect_observed_accuracy_does_not_imply_perfect_population_accuracy():
    from mindscape.coding.statistics import wilson_interval

    lower, upper = wilson_interval(100, 100)
    assert 0.96 < lower < 0.97
    assert upper == pytest.approx(1)
