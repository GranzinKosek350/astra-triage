"""Gradient reversal and the site term of Eq. (3).

Ref: Sec. 2.3.0.3, "the gradient reversal term prevents the merged representation
from incorporating the encoding of a site, thus the adapter does not turn into a
site classifier"; Sec. 2.3.0.4, L_site.

The reversal sits between the merged representation and the site head. The head
still minimises its own cross-entropy, so it stays an honest probe; the sign flip
hands the encoder the opposite gradient, which pushes site identity out of the
representation. The strength follows the usual ramp from zero so that the site head
has something to fit before the encoder is punished.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch.nn import functional as F


class _Reversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, value: torch.Tensor, strength: float) -> torch.Tensor:
        ctx.strength = strength
        return value.view_as(value)

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.strength * gradient, None


def reverse_gradient(value: torch.Tensor, strength: float) -> torch.Tensor:
    reversed_value: torch.Tensor = _Reversal.apply(value, strength)  # type: ignore[no-untyped-call]
    return reversed_value


def ramp(step: int, total: int, maximum: float = 1.0) -> float:
    """The 2/(1+exp(-10p))-1 schedule, reaching the requested maximum at p = 1."""
    if total <= 0:
        return maximum
    progress = min(1.0, max(0.0, step / total))
    if progress >= 1.0:
        return maximum
    return maximum * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)


def site_loss(site_logits: torch.Tensor, site: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(site_logits, site)
