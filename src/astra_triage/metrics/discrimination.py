"""Discrimination: AUROC with its DeLong variance, and AUPRC.

Ref: Table 2, Table 5, Table 6, and the footnote "All differences are paired
bootstrap estimates over 2,000 resamples."

The area is computed from mid-ranks, which is the Mann-Whitney form and is exact
under ties; the variance is the DeLong placement-value form, which needs the
pairwise matrix and is therefore only used where the cohort is small enough to
afford it. AUPRC is the step-wise average precision, so a model that ranks the few
events well is not rewarded for the shape of the precision curve between them.
"""

from __future__ import annotations

import math

import numpy as np
import torch


def _as_arrays(scores: torch.Tensor, labels: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    score_array = scores.detach().to(torch.float64).cpu().numpy().reshape(-1)
    label_array = labels.detach().to(torch.float64).cpu().numpy().reshape(-1)
    if score_array.shape != label_array.shape:
        raise ValueError("scores and labels must align")
    if not np.isin(label_array, [0.0, 1.0]).all():
        raise ValueError("the endpoint is binary")
    return score_array, label_array


def auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    score_array, label_array = _as_arrays(scores, labels)
    positives = int(label_array.sum())
    negatives = int(label_array.size - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC needs both classes")
    order = np.argsort(score_array, kind="mergesort")
    ordered = score_array[order]
    ranks = np.empty(ordered.size, dtype=np.float64)
    index = 0
    while index < ordered.size:
        stop = index
        while stop + 1 < ordered.size and ordered[stop + 1] == ordered[index]:
            stop += 1
        ranks[order[index : stop + 1]] = 0.5 * (index + stop) + 1.0
        index = stop + 1
    positive_rank_sum = ranks[label_array == 1.0].sum()
    statistic = positive_rank_sum - positives * (positives + 1) / 2.0
    return float(statistic / (positives * negatives))


def auroc_variance(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """DeLong variance of the area, from the two placement-value samples."""
    score_array, label_array = _as_arrays(scores, labels)
    positive = score_array[label_array == 1.0]
    negative = score_array[label_array == 0.0]
    if positive.size < 2 or negative.size < 2:
        raise ValueError("the DeLong variance needs at least two of each class")
    difference = positive[:, None] - negative[None, :]
    placement = (difference > 0).astype(np.float64) + 0.5 * (difference == 0).astype(np.float64)
    structural = placement.mean(axis=1)
    structural_negative = placement.mean(axis=0)
    return float(structural.var(ddof=1) / positive.size + structural_negative.var(ddof=1) / negative.size)


def auroc_interval(scores: torch.Tensor, labels: torch.Tensor, z: float = 1.959964) -> tuple[float, float]:
    area = auroc(scores, labels)
    standard_error = math.sqrt(max(auroc_variance(scores, labels), 0.0))
    return max(0.0, area - z * standard_error), min(1.0, area + z * standard_error)


def auroc_standard_error(scores: torch.Tensor, labels: torch.Tensor) -> float:
    return math.sqrt(max(auroc_variance(scores, labels), 0.0))


def hanley_mcneil_variance(area: float, positives: int, negatives: int) -> float:
    """Closed-form variance from the area and the class counts alone.

    Used where only printed counts are available; the DeLong form is preferred
    whenever the scores themselves are at hand.
    """
    if positives < 1 or negatives < 1:
        raise ValueError("both classes must be present")
    q1 = area / (2.0 - area)
    q2 = 2.0 * area * area / (1.0 + area)
    numerator = area * (1.0 - area)
    numerator += (positives - 1) * (q1 - area * area)
    numerator += (negatives - 1) * (q2 - area * area)
    return float(numerator / (positives * negatives))


def auprc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    score_array, label_array = _as_arrays(scores, labels)
    positives = float(label_array.sum())
    if positives == 0.0:
        raise ValueError("AUPRC needs at least one event")
    order = np.argsort(-score_array, kind="mergesort")
    ordered_labels = label_array[order]
    true_positive = np.cumsum(ordered_labels)
    false_positive = np.cumsum(1.0 - ordered_labels)
    precision = true_positive / np.maximum(true_positive + false_positive, 1.0)
    recall = true_positive / positives
    increments = np.diff(np.concatenate([[0.0], recall]))
    return float((increments * precision).sum())


def prevalence(labels: torch.Tensor) -> float:
    return float(labels.detach().to(torch.float64).mean())
