"""Ablation surgery and the additive bound.

Ref: Table 3, all eight rows, and the note "The last row is the additive bound
obtained by averaging the two single-stream model scores; the full model exceeding
that bound is the interaction evidence."

Removing a component happens at the input or in the configuration, never by zeroing
a learned tensor after the fact. Removing a stream marks it structurally absent for
every case, which is the same path the external cohort takes when it carries no
ultrasound. The additive bound is the rank average of the two single-stream scores:
averaging two areas is not the area of an averaged score, so the bound has to be
built from scores and then scored once.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from astra_triage.metrics.discrimination import auroc
from astra_triage.streams.case import Episode


def drop_stream(episodes: list[Episode], stream: str) -> list[Episode]:
    if stream == "ultrasound":
        return [replace(e, ultrasound=(), ultrasound_recorded=False) for e in episodes]
    if stream == "markers":
        return [replace(e, markers=(), markers_recorded=False) for e in episodes]
    raise KeyError(f"unknown stream: {stream}")


def rank_average(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Average of the two scores after each is mapped to its within-cohort rank."""
    if first.shape != second.shape:
        raise ValueError("the two score vectors must align")
    if first.numel() < 2:
        raise ValueError("a rank average needs at least two episodes")

    def ranks(values: torch.Tensor) -> torch.Tensor:
        order = torch.argsort(values, stable=True)
        positions = torch.empty_like(order, dtype=torch.float32)
        positions[order] = torch.arange(values.numel(), dtype=torch.float32)
        return positions / (values.numel() - 1)

    return 0.5 * (ranks(first) + ranks(second))


def additive_bound(
    without_ultrasound: torch.Tensor,
    without_markers: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    bound = rank_average(without_ultrasound, without_markers)
    return bound, auroc(bound, labels)


def interaction_gain(full: float, bound: float) -> float:
    """Percentage points the full model adds over the bound."""
    return 100.0 * (full - bound)
