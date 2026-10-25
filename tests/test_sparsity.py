"""Unobserved stream-days and the four bands."""

from __future__ import annotations

import pytest

from astra_triage.observation.sparsity import (
    STRATUM_NAMES,
    cohort_shares,
    day_absence,
    stratum,
)
from support import make_episode


def test_day_count_runs_to_the_index_day() -> None:
    episode = make_episode(1, 0, ultrasound=(10.0,), markers=(12.0,), index_hours=72.0)
    absence = day_absence(episode)
    assert absence.days_total == 4


def test_observed_days_are_counted_once_per_stream() -> None:
    episode = make_episode(1, 0, ultrasound=(10.0, 11.5, 40.0), markers=(12.0, 13.0), index_hours=60.0)
    absence = day_absence(episode)
    assert absence.ultrasound_days_observed == 2
    assert absence.marker_days_observed == 1


def test_absent_stream_is_fully_unobserved() -> None:
    episode = make_episode(1, 0, ultrasound=(), ultrasound_recorded=False, index_hours=48.0)
    absence = day_absence(episode)
    assert absence.ultrasound_fraction == 1.0


def test_pooled_fraction_averages_the_two_streams() -> None:
    episode = make_episode(1, 0, ultrasound=(10.0,), markers=(12.0, 40.0), index_hours=48.0)
    absence = day_absence(episode)
    assert absence.pooled_fraction == pytest.approx(0.5)


def test_stratum_edges_are_closed_at_the_bottom() -> None:
    assert stratum(0.19) == "low"
    assert stratum(0.20) == "medium"
    assert stratum(0.45) == "high"
    assert stratum(0.61) == "very_high"
    assert stratum(1.0) == "very_high"


def test_stratum_names_are_the_four_bands() -> None:
    assert STRATUM_NAMES == ("low", "medium", "high", "very_high")


def test_shares_are_percentages() -> None:
    rows = [make_episode(i, i % 2, ultrasound=(10.0,), index_hours=48.0) for i in range(4)]
    shares = cohort_shares(rows)
    assert 0.0 <= shares["days_without_ultrasound"] <= 100.0


def test_empty_cohort_is_rejected() -> None:
    with pytest.raises(ValueError):
        cohort_shares([])
