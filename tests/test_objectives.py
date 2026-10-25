"""Eq. (3), its three terms and the gradient reversal."""

from __future__ import annotations

import math

import torch

from astra_triage.model.astra import build_astra
from astra_triage.objectives.calibration import bin_table, decile_penalty
from astra_triage.objectives.composite import LossWeights, binary_cross_entropy, composite_loss
from astra_triage.objectives.effective_number import (
    class_weights,
    effective_number,
    positive_weight,
)
from astra_triage.objectives.site_loss import ramp, reverse_gradient
from astra_triage.streams.case import build_batch
from support import make_episode


def test_effective_number_saturates() -> None:
    assert effective_number(1) < effective_number(100)
    assert effective_number(10000) < 10000


def test_class_weights_sum_to_the_class_count() -> None:
    weights = class_weights([90, 10])
    assert abs(float(weights.sum()) - 2.0) < 1e-5
    assert float(weights[1]) > float(weights[0])


def test_positive_weight_is_above_one_when_the_positive_class_is_rare() -> None:
    assert float(positive_weight(90, 10)) > 1.0


def test_balanced_cross_entropy_is_log_two() -> None:
    logits = torch.zeros(40)
    target = torch.tensor([0.0, 1.0] * 20)
    assert abs(float(binary_cross_entropy(logits, target, 20, 20)) - math.log(2.0)) < 1e-6


def test_imbalanced_cross_entropy_exceeds_the_balanced_value() -> None:
    logits = torch.zeros(100)
    target = torch.tensor([0.0] * 90 + [1.0] * 10)
    assert float(binary_cross_entropy(logits, target, 90, 10)) > math.log(2.0)


def test_calibration_penalty_vanishes_when_bins_are_aligned() -> None:
    probability = torch.tensor([0.05] * 100 + [0.80] * 100)
    target = torch.tensor([0.0] * 95 + [1.0] * 5 + [0.0] * 20 + [1.0] * 80)
    assert float(decile_penalty(probability, target)) < 1e-8


def test_calibration_penalty_grows_when_the_bins_are_reversed() -> None:
    probability = torch.tensor([0.05] * 50 + [0.80] * 50)
    target = torch.tensor([0.0] * 48 + [1.0] * 2 + [0.0] * 10 + [1.0] * 40)
    assert float(decile_penalty(1.0 - probability, target)) > 0.2


def test_bin_table_covers_the_batch() -> None:
    probability = torch.linspace(0.0, 1.0, 101)[1:-1]
    target = (probability > 0.5).to(torch.float32)
    rows = bin_table(probability, target)
    assert abs(sum(row[0] for row in rows) - 1.0) < 1e-5


def test_reversal_flips_the_gradient() -> None:
    value = torch.ones(2, requires_grad=True)
    reverse_gradient(value, 0.25).sum().backward()
    assert value.grad is not None
    assert float(value.grad[0]) == -0.25


def test_ramp_starts_below_one_and_reaches_it() -> None:
    assert ramp(0, 5) < 0.1
    assert abs(ramp(50, 5) - 1.0) < 1e-6


def test_composite_loss_reports_three_terms() -> None:
    model = build_astra()
    batch = build_batch([make_episode(i, i % 2) for i in range(4)], capacity=8)
    output = model(batch)
    breakdown = composite_loss(
        output.logits,
        output.probability,
        batch.label,
        output.site_logits,
        batch.site,
        (2, 2),
        LossWeights(),
    )
    values = breakdown.as_floats()
    assert set(values) == {"total", "bce", "calibration", "site"}
    assert abs(values["total"] - (values["bce"] + 0.5 * values["calibration"] + 0.1 * values["site"])) < 1e-5


def test_zero_weights_leave_only_the_cross_entropy() -> None:
    model = build_astra()
    batch = build_batch([make_episode(i, i % 2) for i in range(4)], capacity=8)
    output = model(batch)
    breakdown = composite_loss(
        output.logits,
        output.probability,
        batch.label,
        output.site_logits,
        batch.site,
        (2, 2),
        LossWeights(lambda_cal=0.0, lambda_site=0.0),
    )
    assert abs(float(breakdown.total.detach()) - float(breakdown.bce.detach())) < 1e-6


def test_negative_penalty_weight_is_rejected() -> None:
    try:
        LossWeights(lambda_cal=-1.0)
    except ValueError:
        return
    raise AssertionError("a negative penalty weight was accepted")
