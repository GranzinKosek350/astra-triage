"""Net benefit and decision curves.

Ref: abstract, "net benefit at the 0.20 threshold was 0.060 per episode against
0.033 for a C-reactive protein rule".

Net benefit is the true-positive share minus the false-positive share weighted by
the odds of the threshold. It is the quantity that makes a rule comparable to a
model on the same episodes, because it charges the rule for the reviews it creates.
The treat-all and treat-none references are carried alongside so a curve can be
read without a second pass over the cohort.
"""

from __future__ import annotations

import torch

from astra_triage.metrics.operating import at_threshold


def net_benefit(sensitivity: float, specificity: float, prevalence: float, threshold: float) -> float:
    if not 0.0 < threshold < 1.0:
        raise ValueError("a decision threshold sits strictly inside (0, 1)")
    true_positive = sensitivity * prevalence
    false_positive = (1.0 - specificity) * (1.0 - prevalence)
    return true_positive - false_positive * threshold / (1.0 - threshold)


def net_benefit_from_scores(probability: torch.Tensor, target: torch.Tensor, threshold: float) -> float:
    point = at_threshold(probability, target, threshold)
    return net_benefit(point.sensitivity, point.specificity, float(target.detach().mean()), threshold)


def curve(
    probability: torch.Tensor,
    target: torch.Tensor,
    thresholds: torch.Tensor | None = None,
) -> list[tuple[float, float, float]]:
    """``(threshold, model, treat_all)`` over the grid."""
    if thresholds is None:
        thresholds = torch.linspace(0.01, 0.60, 60)
    event_rate = float(target.detach().mean())
    rows: list[tuple[float, float, float]] = []
    for value in thresholds.tolist():
        point = at_threshold(probability, target, value)
        rows.append(
            (
                value,
                net_benefit(point.sensitivity, point.specificity, event_rate, value),
                net_benefit(1.0, 0.0, event_rate, value),
            )
        )
    return rows


def net_benefit_advantage(model: torch.Tensor, rule: torch.Tensor, target: torch.Tensor, threshold: float) -> float:
    return net_benefit_from_scores(model, target, threshold) - net_benefit_from_scores(rule, target, threshold)
