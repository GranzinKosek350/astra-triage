"""The three appendix properties, checked for direction rather than magnitude."""

from __future__ import annotations

from astra_triage.properties.counting_estimator import consistency_sweep, selection_bias, size_sweep
from astra_triage.properties.dominance import dominance, nesting_check
from astra_triage.properties.locf_bias import gap_information, slope_sweep


def test_last_observation_needs_the_gap_when_sampling_is_informative() -> None:
    ignorable = gap_information(0.0)
    driven = gap_information(3.0)
    assert ignorable.gap_coefficient > 0.0
    assert driven.gap_coefficient < 0.0
    assert driven.area_gain > ignorable.area_gain


def test_gap_coefficient_moves_monotonically_with_the_dependence() -> None:
    results = slope_sweep()
    coefficients = [result.gap_coefficient for result in results]
    assert coefficients == sorted(coefficients, reverse=True)


def test_observation_share_falls_as_sampling_follows_the_state() -> None:
    results = slope_sweep()
    assert results[-1].observed_share < results[0].observed_share


def test_counting_estimator_is_nearly_unbiased_when_sampling_is_ignorable() -> None:
    result = selection_bias(0.0)
    assert abs(result.bias) < 0.01


def test_counting_estimator_is_biased_when_sampling_follows_the_state() -> None:
    result = selection_bias(3.0)
    assert result.bias > 0.03


def test_within_cohort_calibration_stays_small_in_both_regimes() -> None:
    for result in consistency_sweep():
        assert result.within_cohort_ece < 0.05


def test_bias_does_not_shrink_with_the_cohort() -> None:
    results = size_sweep(2.0)
    spread = max(result.bias for result in results) - min(result.bias for result in results)
    assert spread < 0.02


def test_unconstrained_class_contains_the_admissible_one() -> None:
    assert nesting_check(1.0, episodes=800)


def test_unconstrained_risk_never_exceeds_the_admissible_risk() -> None:
    for slope in (0.0, 1.0, 2.0):
        result = dominance(slope, episodes=800)
        assert result.earned >= -1e-9


def test_dependence_increases_what_the_unconstrained_class_earns() -> None:
    ignorable = dominance(0.0, episodes=1500)
    driven = dominance(2.0, episodes=1500)
    assert driven.earned > ignorable.earned
