"""The two prospective windows.

Ref: Table 9, Sec. 3.9, "The 0.20 threshold is determined from the train dataset and
is not re-assessed in the timeframes".

The two phases are consecutive and cover different children. Neither randomises, so
the comparison between them is a before-and-after description and is labelled as
one. Every quantity in the row is either logged by the instrument or adjudicated
from the record; nothing is inferred from the model's own score.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from astra_triage.metrics.discrimination import auroc


@dataclass(frozen=True)
class DeploymentPhase:
    label: str
    sites: int
    months: int
    episodes: int
    events: int
    scores: torch.Tensor | None = None
    latency_within_five_minutes: float | None = None
    alerts_per_hundred: float | None = None
    reviewed_same_shift: float | None = None
    alert_to_review_minutes: float | None = None
    reading_minutes_per_alert: float | None = None
    time_to_reintervention_hours: float | None = None
    delayed_beyond_48h: float | None = None
    exploration_without_infection: float | None = None
    labels: torch.Tensor | None = None

    @property
    def prevalence(self) -> float:
        if self.episodes == 0:
            raise ValueError("an empty window has no prevalence")
        return 100.0 * self.events / self.episodes

    def discrimination(self) -> float | None:
        if self.scores is None or self.labels is None:
            return None
        return auroc(self.scores, self.labels)

    def row(self) -> dict[str, float | str | None]:
        value = self.discrimination()
        return {
            "phase": self.label,
            "sites": self.sites,
            "months": self.months,
            "episodes": self.episodes,
            "events": self.events,
            "prevalence": self.prevalence,
            "auroc": value,
            "latency_within_five_minutes": self.latency_within_five_minutes,
            "alerts_per_hundred": self.alerts_per_hundred,
            "reviewed_same_shift": self.reviewed_same_shift,
            "alert_to_review_minutes": self.alert_to_review_minutes,
            "reading_minutes_per_alert": self.reading_minutes_per_alert,
            "time_to_reintervention_hours": self.time_to_reintervention_hours,
            "delayed_beyond_48h": self.delayed_beyond_48h,
            "exploration_without_infection": self.exploration_without_infection,
        }


@dataclass
class DeploymentRecord:
    silent: DeploymentPhase
    open_label: DeploymentPhase
    threshold_fixed_at: float
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold_fixed_at < 1.0:
            raise ValueError("the deployment threshold is a probability")
        if self.silent.label == self.open_label.label:
            raise ValueError("the two phases are different windows")

    def table(self) -> list[dict[str, float | str | None]]:
        return [self.silent.row(), self.open_label.row()]

    def before_after(self, key: str) -> tuple[float, float] | None:
        first = self.silent.row().get(key)
        second = self.open_label.row().get(key)
        if first is None or second is None:
            return None
        if not isinstance(first, (int, float)) or not isinstance(second, (int, float)):
            return None
        return float(first), float(second)
