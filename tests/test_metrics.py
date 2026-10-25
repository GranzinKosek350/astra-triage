"""Metrics against independent definitions."""

from __future__ import annotations

import math

import torch

from astra_triage.metrics.agreement import cohen_kappa
from astra_triage.metrics.bootstrap import paired_difference, percentile_interval
from astra_triage.metrics.calibration import (
    brier,
    calibration_slope,
    expected_calibration_error,
    murphy_components,
)
from astra_triage.metrics.discrimination import (
    auprc,
    auroc,
    auroc_interval,
    auroc_variance,
    hanley_mcneil_variance,
)
from astra_triage.metrics.heterogeneity import from_areas, heterogeneity
from astra_triage.metrics.net_benefit import net_benefit, net_benefit_from_scores
from astra_triage.metrics.operating import at_threshold, npv_from, ppv_from, youden_threshold


def test_auroc_matches_the_pairwise_count() -> None:
    scores = torch.tensor([0.1, 0.4, 0.35, 0.8])
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    positive = scores[labels == 1.0]
    negative = scores[labels == 0.0]
    tally = sum(1.0 if p > q else 0.5 if p == q else 0.0 for p in positive for q in negative)
    assert abs(auroc(scores, labels) - tally / 4.0) < 1e-9


def test_auroc_handles_ties() -> None:
    scores = torch.tensor([0.5, 0.5])
    labels = torch.tensor([1.0, 0.0])
    assert abs(auroc(scores, labels) - 0.5) < 1e-9


def test_auroc_rejects_a_single_class() -> None:
    try:
        auroc(torch.tensor([0.1, 0.2]), torch.tensor([1.0, 1.0]))
    except ValueError:
        return
    raise AssertionError("a single class was accepted")


def test_de_long_variance_is_positive_and_interval_brackets() -> None:
    rng = torch.Generator().manual_seed(1)
    scores = torch.rand(300, generator=rng)
    labels = (torch.rand(300, generator=rng) < scores).to(torch.float32)
    variance = auroc_variance(scores, labels)
    low, high = auroc_interval(scores, labels)
    assert variance > 0
    assert low < auroc(scores, labels) < high


def test_hanley_mcneil_shrinks_with_the_sample() -> None:
    small = hanley_mcneil_variance(0.85, 40, 160)
    large = hanley_mcneil_variance(0.85, 400, 1600)
    assert large < small


def test_auprc_equals_the_step_formula() -> None:
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6])
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    expected = 0.5 * 1.0 + 0.5 * (2.0 / 3.0)
    assert abs(auprc(scores, labels) - expected) < 1e-9


def test_auprc_is_one_for_a_perfect_ranking() -> None:
    scores = torch.tensor([0.9, 0.7, 0.6, 0.4])
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
    assert abs(auprc(scores, labels) - 1.0) < 1e-9


def test_brier_and_ece_on_a_known_example() -> None:
    probability = torch.tensor([0.0, 1.0, 0.5, 0.5])
    target = torch.tensor([0.0, 1.0, 1.0, 0.0])
    assert abs(brier(probability, target) - 0.125) < 1e-9
    assert expected_calibration_error(probability, target) < 1e-9


def test_ece_grows_when_the_probabilities_are_shifted() -> None:
    rng = torch.Generator().manual_seed(11)
    probability = torch.rand(2000, generator=rng)
    target = (torch.rand(2000, generator=rng) < probability).to(torch.float32)
    assert expected_calibration_error(torch.clamp(probability - 0.3, min=0.0), target) > 0.15


def test_murphy_components_rebuild_the_brier_score() -> None:
    rng = torch.Generator().manual_seed(3)
    probability = torch.rand(2000, generator=rng)
    target = (torch.rand(2000, generator=rng) < probability).to(torch.float32)
    parts = murphy_components(probability, target)
    assert abs(parts["brier"] - brier(probability, target)) < 2e-3


def test_slope_is_one_for_a_calibrated_predictor() -> None:
    rng = torch.Generator().manual_seed(4)
    signal = torch.randn(5000, generator=rng)
    probability = torch.sigmoid(signal)
    target = (torch.rand(5000, generator=rng) < probability).to(torch.float32)
    _, slope = calibration_slope(probability, target)
    assert abs(slope - 1.0) < 0.15


def test_slope_is_below_one_when_overconfident() -> None:
    rng = torch.Generator().manual_seed(5)
    signal = torch.randn(5000, generator=rng)
    probability = torch.sigmoid(signal)
    target = (torch.rand(5000, generator=rng) < probability).to(torch.float32)
    _, slope = calibration_slope(torch.sigmoid(2.0 * signal), target)
    assert slope < 0.7


def test_operating_point_and_predictive_identities() -> None:
    probability = torch.tensor([0.9, 0.7, 0.15, 0.05])
    target = torch.tensor([1.0, 1.0, 0.0, 0.0])
    point = at_threshold(probability, target, 0.2)
    assert point.sensitivity == 1.0 and point.specificity == 1.0
    prevalence = float(target.mean())
    assert abs(ppv_from(point.sensitivity, point.specificity, prevalence) - 1.0) < 1e-9
    assert abs(npv_from(point.sensitivity, point.specificity, prevalence) - 1.0) < 1e-9


def test_youden_threshold_sits_between_the_classes() -> None:
    probability = torch.tensor([0.1, 0.2, 0.8, 0.9])
    target = torch.tensor([0.0, 0.0, 1.0, 1.0])
    value = youden_threshold(probability, target)
    assert 0.2 <= value <= 0.9


def test_net_benefit_matches_the_counted_form() -> None:
    probability = torch.tensor([0.9, 0.8, 0.7, 0.1, 0.05, 0.02])
    target = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    value = net_benefit_from_scores(probability, target, 0.2)
    expected = 2.0 / 6.0 - (1.0 / 6.0) * 0.25
    assert abs(value - expected) < 1e-6


def test_net_benefit_of_a_perfect_rule_is_the_prevalence() -> None:
    value = net_benefit(1.0, 1.0, 0.25, 0.4)
    assert abs(value - 0.25) < 1e-9


def test_kappa_matches_the_closed_form() -> None:
    first = [1, 1, 0, 0, 1, 0, 1, 0]
    second = [1, 0, 0, 1, 1, 0, 1, 0]
    assert abs(cohen_kappa(first, second) - 0.5) < 1e-9


def test_heterogeneity_is_zero_when_the_sites_agree() -> None:
    result = heterogeneity([0.84, 0.84, 0.84], [0.02, 0.02, 0.02])
    assert result["i_squared"] == 0.0
    assert abs(result["q"]) < 1e-9


def test_heterogeneity_grows_with_the_spread() -> None:
    tight = from_areas([0.84, 0.83, 0.85], [100, 100, 100], [400, 400, 400])
    wide = from_areas([0.70, 0.84, 0.90], [100, 100, 100], [400, 400, 400])
    assert wide["i_squared"] > tight["i_squared"]


def test_percentile_interval_is_deterministic() -> None:
    rng = torch.Generator().manual_seed(7)
    values = torch.rand(120, generator=rng)
    first = percentile_interval(lambda rows: float(values[torch.as_tensor(rows)].mean()), 120, resamples=100, seed=3)
    second = percentile_interval(lambda rows: float(values[torch.as_tensor(rows)].mean()), 120, resamples=100, seed=3)
    assert first.low == second.low and first.high == second.high


def test_paired_difference_is_zero_against_itself() -> None:
    rng = torch.Generator().manual_seed(8)
    values = torch.rand(80, generator=rng)
    metric = lambda rows: float(values[torch.as_tensor(rows)].mean())  # noqa: E731
    difference, low, high, p = paired_difference(metric, metric, 80, resamples=80, seed=4)
    assert difference == 0.0
    assert low <= 0.0 <= high
    assert p > 0.05


def test_interval_of_a_constant_collapses_to_a_point() -> None:
    result = percentile_interval(lambda rows: 0.5, 10, resamples=20, seed=1)
    assert math.isclose(result.low, 0.5) and math.isclose(result.high, 0.5)
