"""One evaluation pass per arm, producing the columns of Table 2.

Ref: Table 2, all eight rows and both panels; footnote on the 0.20 operating point
and the paired bootstrap.

The pass reads a score vector and a label vector and fills discrimination, the
operating point, the calibration block and the interval. It never fills a number the
cohort cannot produce: an arm that cannot read the record reports itself
unavailable with a reason, and its row is written as such.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from astra_triage.comparators.base import ArmResult, Comparator
from astra_triage.endpoints.definition import TRIAGE_THRESHOLD
from astra_triage.metrics.bootstrap import RESAMPLES, paired_difference, percentile_interval
from astra_triage.metrics.calibration import (
    brier,
    calibration_slope,
    expected_calibration_error,
)
from astra_triage.metrics.discrimination import auprc, auroc
from astra_triage.metrics.operating import at_threshold
from astra_triage.streams.case import Episode
from astra_triage.training.loop import predict


@dataclass(frozen=True)
class ArmReport:
    name: str
    episodes: int
    events: int
    auroc: float
    auroc_low: float
    auroc_high: float
    auprc: float
    sensitivity: float
    specificity: float
    ppv: float
    npv: float
    brier: float
    ece: float
    slope: float
    available: bool = True
    reason: str | None = None

    def row(self) -> dict[str, float | str | bool | None]:
        """The table row, with an unmeasured column written as null rather than NaN."""
        return {
            "arm": self.name,
            "episodes": float(self.episodes),
            "events": float(self.events),
            "auroc": _finite(self.auroc),
            "auroc_low": _finite(self.auroc_low),
            "auroc_high": _finite(self.auroc_high),
            "auprc": _finite(self.auprc),
            "sensitivity": _finite(self.sensitivity),
            "specificity": _finite(self.specificity),
            "ppv": _finite(self.ppv),
            "npv": _finite(self.npv),
            "brier": _finite(self.brier),
            "ece": _finite(self.ece),
            "slope": _finite(self.slope),
            "available": self.available,
            "reason": self.reason or "",
        }

    @property
    def summary(self) -> str:
        if not self.available:
            return f"{self.name}: not run ({self.reason})"
        return (
            f"{self.name}: AUROC {self.auroc:.3f} [{self.auroc_low:.3f}, {self.auroc_high:.3f}], "
            f"ECE {self.ece:.3f}, slope {self.slope:.3f}"
        )


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def unavailable_report(name: str, episodes: int, reason: str) -> ArmReport:
    return ArmReport(
        name=name,
        episodes=episodes,
        events=0,
        auroc=float("nan"),
        auroc_low=float("nan"),
        auroc_high=float("nan"),
        auprc=float("nan"),
        sensitivity=float("nan"),
        specificity=float("nan"),
        ppv=float("nan"),
        npv=float("nan"),
        brier=float("nan"),
        ece=float("nan"),
        slope=float("nan"),
        available=False,
        reason=reason,
    )


def scores_of(arm: nn.Module | Comparator, episodes: list[Episode]) -> tuple[torch.Tensor, str | None]:
    """Every arm exposes ``score``; only a bare module falls back to the batch path."""
    scorer = getattr(arm, "score", None)
    if callable(scorer):
        result: ArmResult = scorer(episodes)
        if not result.available:
            return result.scores, result.reason or "arm unavailable"
        return result.scores, None
    if not isinstance(arm, nn.Module):
        raise TypeError("an arm needs either a score method or a module body")
    return predict(arm, episodes), None


def evaluate(
    name: str,
    scores: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = TRIAGE_THRESHOLD,
    resamples: int = RESAMPLES,
    seed: int = 20260825,
) -> ArmReport:
    if torch.isnan(scores).any():
        raise ValueError("a scored arm produced a non-finite probability")
    point = at_threshold(scores, labels, threshold)
    interval = percentile_interval(
        lambda rows: auroc(scores[torch.as_tensor(rows)], labels[torch.as_tensor(rows)]),
        int(labels.numel()),
        resamples=resamples,
        seed=seed,
    )
    slope = calibration_slope(scores, labels)[1]
    return ArmReport(
        name=name,
        episodes=int(labels.numel()),
        events=int(labels.sum()),
        auroc=auroc(scores, labels),
        auroc_low=interval.low,
        auroc_high=interval.high,
        auprc=auprc(scores, labels),
        sensitivity=point.sensitivity,
        specificity=point.specificity,
        ppv=point.ppv,
        npv=point.npv,
        brier=brier(scores, labels),
        ece=expected_calibration_error(scores, labels),
        slope=slope,
    )


def evaluate_arm(
    arm: nn.Module | Comparator,
    episodes: list[Episode],
    name: str | None = None,
    resamples: int = RESAMPLES,
    seed: int = 20260825,
) -> ArmReport:
    label = name or getattr(arm, "name", type(arm).__name__)
    scores, reason = scores_of(arm, episodes)
    if reason is not None:
        return unavailable_report(str(label), len(episodes), reason)
    targets = torch.tensor([float(e.label) for e in episodes], dtype=torch.float32)
    return evaluate(str(label), scores, targets, resamples=resamples, seed=seed)


def evaluate_table(
    arms: Sequence[nn.Module | Comparator],
    episodes: list[Episode],
    resamples: int = RESAMPLES,
    seed: int = 20260825,
) -> list[ArmReport]:
    return [evaluate_arm(arm, episodes, resamples=resamples, seed=seed) for arm in arms]


def paired_delta(
    first: torch.Tensor,
    second: torch.Tensor,
    labels: torch.Tensor,
    resamples: int = RESAMPLES,
    seed: int = 20260825,
) -> tuple[float, float, float, float]:
    """Paired bootstrap of an AUROC difference on the same episode draws."""
    total = int(labels.numel())

    def scored(scores: torch.Tensor) -> Callable[[np.ndarray], float]:
        def metric(rows: np.ndarray) -> float:
            index = torch.as_tensor(rows)
            return auroc(scores[index], labels[index])

        return metric

    return paired_difference(scored(first), scored(second), total, resamples=resamples, seed=seed)
