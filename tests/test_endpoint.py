"""Endpoint, adjudication and banding."""

from __future__ import annotations

import pytest

from astra_triage.endpoints.adjudication import Decision, adjudicate, outcome_from
from astra_triage.endpoints.definition import (
    Outcome,
    OutcomeKind,
    TriageBand,
    band,
    index_visible,
    label,
)


def test_inside_window_is_positive() -> None:
    outcome = Outcome(kind=OutcomeKind.REINTERVENTION, reintervention_hours=48.0)
    assert label(outcome) == 1


def test_window_is_closed_on_day_seven() -> None:
    assert label(Outcome(kind=OutcomeKind.REINTERVENTION, reintervention_hours=168.0)) == 1
    assert label(Outcome(kind=OutcomeKind.REINTERVENTION, reintervention_hours=168.5)) == 0


def test_antibiotic_escalation_is_a_non_event() -> None:
    outcome = Outcome(kind=OutcomeKind.ANTIBIOTIC_ESCALATION, antibiotic_escalated=True)
    assert label(outcome) == 0


def test_reintervention_without_a_time_is_rejected() -> None:
    with pytest.raises(ValueError):
        Outcome(kind=OutcomeKind.REINTERVENTION)


def test_non_surgical_outcome_cannot_carry_a_time() -> None:
    with pytest.raises(ValueError):
        Outcome(kind=OutcomeKind.UNEVENTFUL, reintervention_hours=10.0)


def test_index_visibility_is_inclusive() -> None:
    assert index_visible(40.0, 40.0)
    assert not index_visible(40.5, 40.0)


def test_bands_follow_the_two_cut_points() -> None:
    assert band(0.05) is TriageBand.LOW
    assert band(0.20) is TriageBand.BORDERLINE
    assert band(0.65) is TriageBand.HIGH


def test_concordant_read_needs_no_arbiter() -> None:
    record = adjudicate(Decision.REINTERVENTION, Decision.REINTERVENTION)
    assert record.decision is Decision.REINTERVENTION
    assert record.concordant


def test_discordant_read_needs_an_arbiter() -> None:
    with pytest.raises(ValueError):
        adjudicate(Decision.REINTERVENTION, Decision.NO_REINTERVENTION)
    record = adjudicate(Decision.REINTERVENTION, Decision.NO_REINTERVENTION, Decision.NO_REINTERVENTION)
    assert record.decision is Decision.NO_REINTERVENTION


def test_outcome_from_decision() -> None:
    record = adjudicate(Decision.NO_REINTERVENTION, Decision.NO_REINTERVENTION)
    outcome = outcome_from(record, None, escalated=True)
    assert outcome.kind is OutcomeKind.ANTIBIOTIC_ESCALATION
