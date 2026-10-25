"""Unobserved stream-days, and the four sparsity strata of Table 6 panel B.

Ref: Table 1 ("Days without ultrasound, %"), Table 6 panel B, abstract.

A stream-day is one calendar day of one episode crossed with one stream. A day is
observed for a stream when at least one reading of that stream is stamped on it,
counting from admission to the index day inclusive. The article reports the
unobserved share as a percentage of stream-days and stratifies on it, so the same
arithmetic is used for the cohort description and for the strata. Two boundaries
are settled here because the article prints only labelled bands: the lower edge of
a band belongs to that band, and the very-high band also holds the closed upper
edge at 1.0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from astra_triage.streams.case import Episode

STRATUM_EDGES: tuple[float, ...] = (0.20, 0.40, 0.60)
STRATUM_NAMES: tuple[str, ...] = ("low", "medium", "high", "very_high")


@dataclass(frozen=True)
class DayAbsence:
    days_total: int
    ultrasound_days_observed: int
    marker_days_observed: int
    ultrasound_recorded: bool
    markers_recorded: bool

    @property
    def ultrasound_fraction(self) -> float:
        if self.days_total == 0 or not self.ultrasound_recorded:
            return 1.0
        return 1.0 - self.ultrasound_days_observed / self.days_total

    @property
    def marker_fraction(self) -> float:
        if self.days_total == 0 or not self.markers_recorded:
            return 1.0
        return 1.0 - self.marker_days_observed / self.days_total

    @property
    def pooled_fraction(self) -> float:
        streams = int(self.ultrasound_recorded) + int(self.markers_recorded)
        if streams == 0 or self.days_total == 0:
            return 1.0
        observed = self.ultrasound_days_observed * int(self.ultrasound_recorded)
        observed += self.marker_days_observed * int(self.markers_recorded)
        return 1.0 - observed / (streams * self.days_total)


def _observed_days(times: list[float], last_day: int) -> int:
    days = {min(math.floor(t / 24.0), last_day) for t in times}
    return len(days)


def day_absence(episode: Episode) -> DayAbsence:
    """Calendar-day coverage of both streams up to the index time."""
    last_day = max(0, math.floor(episode.index_hours / 24.0))
    ultrasound = [o.tau_hours for o in episode.ultrasound if o.tau_hours <= episode.index_hours]
    markers = [o.tau_hours for o in episode.markers if o.tau_hours <= episode.index_hours]
    return DayAbsence(
        days_total=last_day + 1,
        ultrasound_days_observed=_observed_days(ultrasound, last_day),
        marker_days_observed=_observed_days(markers, last_day),
        ultrasound_recorded=episode.ultrasound_recorded,
        markers_recorded=episode.markers_recorded,
    )


def stratum(fraction: float) -> str:
    """Band a pooled unobserved-stream-day fraction."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("a fraction of stream-days lies in [0, 1]")
    for edge, name in zip(STRATUM_EDGES, STRATUM_NAMES[:-1], strict=True):
        if fraction < edge:
            return name
    return STRATUM_NAMES[-1]


def cohort_shares(episodes: list[Episode]) -> dict[str, float]:
    """Mean per-stream unobserved share over a set of episodes, as percentages."""
    if not episodes:
        raise ValueError("shares need at least one episode")
    rows = [day_absence(e) for e in episodes]
    return {
        "days_without_ultrasound": 100.0 * sum(r.ultrasound_fraction for r in rows) / len(rows),
        "days_without_marker_panel": 100.0 * sum(r.marker_fraction for r in rows) / len(rows),
    }
