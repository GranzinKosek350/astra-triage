"""The training objective, Eq. (3).

Ref: Sec. 2.3.0.4, Eq. (3):

    L = L_BCE + lambda_1 L_cal + lambda_2 L_site

L_BCE is the class-weighted binary cross-entropy with effective-number weights,
L_cal is the batch risk-decile penalty, and L_site is the gradient-reversal site
term. Both penalty weights are supplied by the caller: the article fits them on the
derivation cohort alone, so they are a property of the run and not of the module.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from astra_triage.objectives.calibration import decile_penalty
from astra_triage.objectives.effective_number import DEFAULT_BETA, positive_weight
from astra_triage.objectives.site_loss import site_loss


@dataclass(frozen=True)
class LossWeights:
    lambda_cal: float = 0.50
    lambda_site: float = 0.10
    beta: float = DEFAULT_BETA
    calibration_bins: int = 10

    def __post_init__(self) -> None:
        if self.lambda_cal < 0 or self.lambda_site < 0:
            raise ValueError("penalty weights cannot be negative")


@dataclass(frozen=True)
class LossBreakdown:
    total: torch.Tensor
    bce: torch.Tensor
    calibration: torch.Tensor
    site: torch.Tensor

    def as_floats(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach()),
            "bce": float(self.bce.detach()),
            "calibration": float(self.calibration.detach()),
            "site": float(self.site.detach()),
        }


def binary_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    negatives: int,
    positives: int,
    beta: float = DEFAULT_BETA,
) -> torch.Tensor:
    weight = positive_weight(negatives, positives, beta).to(logits.device)
    loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=weight)
    return loss


def composite_loss(
    logits: torch.Tensor,
    probability: torch.Tensor,
    target: torch.Tensor,
    site_logits: torch.Tensor,
    site: torch.Tensor,
    counts: tuple[int, int],
    weights: LossWeights,
) -> LossBreakdown:
    bce = binary_cross_entropy(logits, target, counts[0], counts[1], weights.beta)
    calibration = decile_penalty(probability, target, weights.calibration_bins)
    site_term = site_loss(site_logits, site)
    total = bce + weights.lambda_cal * calibration + weights.lambda_site * site_term
    return LossBreakdown(total=total, bce=bce, calibration=calibration, site=site_term)
