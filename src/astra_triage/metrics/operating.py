"""Operating characteristics at the pre-specified 0.20 threshold.

Ref: Table 2 footnote, "PPV and NPV, positive and negative predictive value at the
pre-specified 0.20 threshold."

Sensitivity and specificity are read off the cohort. PPV and NPV are reported twice:
once from the cohort counts and once from the identity that ties them to
sensitivity, specificity and prevalence. The second form is what makes the printed
table checkable without the records, so both live here and both are used.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from astra_triage.endpoints.definition import TRIAGE_THRESHOLD


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float
    sensitivity: float
    specificity: float
    ppv: float
    npv: float
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    @property
    def positives(self) -> int:
        return self.true_positive + self.false_negative

    @property
    def negatives(self) -> int:
        return self.true_negative + self.false_positive


def at_threshold(
    probability: torch.Tensor, target: torch.Tensor, threshold: float = TRIAGE_THRESHOLD
) -> OperatingPoint:
    predicted = probability.detach() >= threshold
    observed = target.detach() >= 0.5
    true_positive = int((predicted & observed).sum())
    false_positive = int((predicted & ~observed).sum())
    true_negative = int((~predicted & ~observed).sum())
    false_negative = int((~predicted & observed).sum())
    sensitivity = true_positive / max(true_positive + false_negative, 1)
    specificity = true_negative / max(true_negative + false_positive, 1)
    ppv = true_positive / max(true_positive + false_positive, 1)
    npv = true_negative / max(true_negative + false_negative, 1)
    return OperatingPoint(
        threshold=threshold,
        sensitivity=sensitivity,
        specificity=specificity,
        ppv=ppv,
        npv=npv,
        true_positive=true_positive,
        false_positive=false_positive,
        true_negative=true_negative,
        false_negative=false_negative,
    )


def ppv_from(sensitivity: float, specificity: float, prevalence: float) -> float:
    """Predictive value implied by the operating point and the event rate."""
    numerator = sensitivity * prevalence
    denominator = numerator + (1.0 - specificity) * (1.0 - prevalence)
    if denominator <= 0:
        raise ValueError("an empty predicted-positive set has no predictive value")
    return numerator / denominator


def npv_from(sensitivity: float, specificity: float, prevalence: float) -> float:
    numerator = specificity * (1.0 - prevalence)
    denominator = numerator + (1.0 - sensitivity) * prevalence
    if denominator <= 0:
        raise ValueError("an empty predicted-negative set has no predictive value")
    return numerator / denominator


def youden_threshold(probability: torch.Tensor, target: torch.Tensor) -> float:
    """The threshold that maximises sensitivity + specificity - 1.

    Only used for descriptive reporting; every printed operating point is taken at
    0.20, which is fixed before the cohort is read.
    """
    scores, labels = probability.detach(), target.detach()
    order = torch.argsort(scores, descending=True)
    ordered_scores = scores[order]
    ordered_labels = labels[order]
    positives = float(ordered_labels.sum())
    negatives = float(len(ordered_labels) - positives)
    true_positive = torch.cumsum(ordered_labels, dim=0)
    false_positive = torch.cumsum(1.0 - ordered_labels, dim=0)
    youden = true_positive / positives + (negatives - false_positive) / negatives - 1.0
    best = int(torch.argmax(youden))
    return float(ordered_scores[best])
