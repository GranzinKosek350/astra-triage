"""Class weights from the effective-number principle.

Ref: Sec. 2.3.0.4, "The class weights adhered to the effective-number principle."

With a positive share near one in eight, a plain inverse-frequency weight would
over-correct the rare class. The effective number of samples of a class with n
members is (1 - beta^n) / (1 - beta), so the weight 1 / E_n saturates as n grows and
the two classes stay comparable in gradient magnitude.
"""

from __future__ import annotations

import torch

DEFAULT_BETA: float = 0.9999


def effective_number(count: int, beta: float = DEFAULT_BETA) -> float:
    if count <= 0:
        raise ValueError("a class with no members has no effective number")
    if not 0.0 < beta < 1.0:
        raise ValueError("beta sits strictly between zero and one")
    return (1.0 - beta**count) / (1.0 - beta)


def class_weights(counts: list[int], beta: float = DEFAULT_BETA) -> torch.Tensor:
    """Weights on the effective-number scale, normalised to mean one."""
    if len(counts) < 2:
        raise ValueError("two classes are needed for a class weighting")
    raw = torch.tensor([1.0 / effective_number(c, beta) for c in counts], dtype=torch.float32)
    return raw * (len(counts) / raw.sum())


def positive_weight(negatives: int, positives: int, beta: float = DEFAULT_BETA) -> torch.Tensor:
    """The scalar a binary cross-entropy expects on its positive term."""
    weights = class_weights([negatives, positives], beta)
    return weights[1] / weights[0]
