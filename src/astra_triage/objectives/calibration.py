"""Batch calibration penalty on risk deciles.

Ref: Sec. 2.3.0.4, "L_cal imposes a penalty for squared deviation from predicted
probability to observed frequencies in the risk deciles for every training batch."

The bins are equal-count bins of the batch's own predicted probabilities, so the
penalty is a reliability term rather than a fixed-grid histogram: every bin holds
the same number of episodes and the deviation of the bin mean from the bin event
rate is squared. Bin membership is a step function of the prediction and carries no
gradient; the bin means do, which is where the penalty acts.
"""

from __future__ import annotations

import torch

DECILES: int = 10


def decile_penalty(
    probability: torch.Tensor,
    target: torch.Tensor,
    bins: int = DECILES,
) -> torch.Tensor:
    """Mean squared bin-wise deviation, weighted by bin occupancy."""
    if probability.shape != target.shape:
        raise ValueError("predictions and labels must align")
    if probability.numel() < bins:
        return torch.zeros((), dtype=probability.dtype, device=probability.device)
    edges = torch.quantile(probability.detach(), torch.linspace(0.0, 1.0, bins + 1)[1:-1].to(probability.device))
    membership = torch.bucketize(probability, edges)
    total = torch.zeros((), dtype=probability.dtype, device=probability.device)
    for index in range(bins):
        live = (membership == index).to(probability.dtype)
        count = live.sum()
        if float(count) == 0.0:
            continue
        mean_prediction = (probability * live).sum() / count
        mean_event = (target * live).sum() / count
        share = count / probability.numel()
        total = total + share * (mean_prediction - mean_event) ** 2
    return total


def bin_table(
    probability: torch.Tensor,
    target: torch.Tensor,
    bins: int = DECILES,
) -> list[tuple[float, float, float]]:
    """``(share, mean prediction, event rate)`` per occupied bin, for reporting."""
    if probability.shape != target.shape:
        raise ValueError("predictions and labels must align")
    edges = torch.quantile(probability.detach(), torch.linspace(0.0, 1.0, bins + 1)[1:-1].to(probability.device))
    membership = torch.bucketize(probability, edges)
    rows: list[tuple[float, float, float]] = []
    for index in range(bins):
        live = (membership == index).to(probability.dtype)
        count = float(live.sum())
        if count == 0.0:
            continue
        rows.append(
            (
                count / probability.numel(),
                float((probability * live).sum() / count),
                float((target * live).sum() / count),
            )
        )
    return rows
