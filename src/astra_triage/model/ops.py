"""Mask-aware reductions shared by the fusion, gating and probe modules."""

from __future__ import annotations

import torch


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax over the live entries; a row with nothing live returns zeros."""
    filled = scores.masked_fill(~mask, float("-inf"))
    largest = filled.max(dim=dim, keepdim=True).values
    largest = torch.where(torch.isfinite(largest), largest, torch.zeros_like(largest))
    weights = torch.exp(filled - largest) * mask.to(scores.dtype)
    total = weights.sum(dim=dim, keepdim=True)
    return torch.where(total > 0, weights / torch.clamp(total, min=1e-12), weights)


def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int = -2) -> torch.Tensor:
    """Mean over the live entries, blind to how many entries there are."""
    weights = mask.to(values.dtype).unsqueeze(-1)
    total = torch.clamp(weights.sum(dim=dim, keepdim=True), min=1.0)
    summed = (values * weights).sum(dim=dim, keepdim=True)
    return (summed / total).squeeze(dim)


def masked_last(values: torch.Tensor, mask: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    """The reading with the largest timestamp; the first slot when nothing is live."""
    newest = torch.where(mask, times, torch.full_like(times, -1e9))
    position = torch.argmax(newest, dim=-1)
    gathered = torch.gather(values, 1, position.view(-1, 1, 1).expand(-1, 1, values.shape[-1]))
    return gathered.squeeze(1)
