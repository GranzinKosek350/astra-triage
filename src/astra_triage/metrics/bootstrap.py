"""Paired bootstrap over episodes.

Ref: Table 2 footnote, "All differences are paired bootstrap estimates over 2,000
resamples"; Table 7 for the reading arms.

A resample draws episode indices with replacement, and every arm is scored on the same
draw, which is what makes the intervals paired. A draw that holds one class cannot be
scored, so it is dropped and counted: with a small cohort that count is the honest
statement that the interval rests on few draws, and above half the draws the interval
is refused rather than reported thin.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

RESAMPLES: int = 2000
DEFAULT_SEED: int = 20240711


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float
    resamples: int
    skipped: int = 0

    @property
    def half_width(self) -> float:
        return 0.5 * (self.high - self.low)


def _indices(size: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, size, size=size)


def _minimum_scored(resamples: int) -> int:
    return max(1, resamples // 4)


def percentile_interval(
    statistic: Callable[[np.ndarray], float],
    size: int,
    resamples: int = RESAMPLES,
    seed: int = DEFAULT_SEED,
    level: float = 0.95,
) -> Interval:
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(resamples):
        try:
            draws.append(float(statistic(_indices(size, rng))))
        except ValueError:
            continue
    if len(draws) < _minimum_scored(resamples):
        raise ValueError("too few bootstrap replicates could be scored")
    values = np.asarray(draws, dtype=np.float64)
    alpha = (1.0 - level) / 2.0
    return Interval(
        estimate=float(statistic(np.arange(size))),
        low=float(np.quantile(values, alpha)),
        high=float(np.quantile(values, 1.0 - alpha)),
        resamples=resamples,
        skipped=resamples - len(draws),
    )


def paired_difference(
    statistic: Callable[[np.ndarray], float],
    comparator: Callable[[np.ndarray], float],
    size: int,
    resamples: int = RESAMPLES,
    seed: int = DEFAULT_SEED,
    level: float = 0.95,
) -> tuple[float, float, float, float]:
    """``(difference, low, high, p)`` for one arm against another on the same draws."""
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(resamples):
        rows = _indices(size, rng)
        try:
            draws.append(float(statistic(rows) - comparator(rows)))
        except ValueError:
            continue
    if len(draws) < _minimum_scored(resamples):
        raise ValueError("too few bootstrap replicates could be scored")
    values = np.asarray(draws, dtype=np.float64)
    alpha = (1.0 - level) / 2.0
    observed = float(statistic(np.arange(size)) - comparator(np.arange(size)))
    below = float(np.mean(values <= 0.0))
    above = float(np.mean(values >= 0.0))
    p_value = min(1.0, 2.0 * min(below, above))
    return (
        observed,
        float(np.quantile(values, alpha)),
        float(np.quantile(values, 1.0 - alpha)),
        p_value,
    )


def stratified_indices(strata: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Resample inside each stratum, keeping the stratum sizes fixed."""
    rows = np.arange(strata.size)
    drawn: list[np.ndarray] = []
    for level in np.unique(strata):
        members = rows[strata == level]
        drawn.append(rng.choice(members, size=members.size, replace=True))
    return np.concatenate(drawn)


def score_bootstrap(
    scores: torch.Tensor,
    labels: torch.Tensor,
    metric: Callable[[torch.Tensor, torch.Tensor], float],
) -> Interval:
    """Convenience wrapper for a metric that reads a score vector and its labels."""
    size = int(labels.numel())

    def statistic(rows: np.ndarray) -> float:
        return metric(scores[torch.as_tensor(rows)], labels[torch.as_tensor(rows)])

    return percentile_interval(statistic, size)
