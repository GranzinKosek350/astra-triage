"""Two-reader adjudication with an independent third.

Ref: Table 8 footnote, "Adjudication was performed by two clinicians ... with
disagreements resolved by a third; both adjudicators were blind to the score."

The adjudicator object deliberately carries no model output: the record of a
blind read is the pair of decisions and, when they differ, the arbiter's call.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from astra_triage.endpoints.definition import Outcome


class Decision(StrEnum):
    REINTERVENTION = "reintervention"
    NO_REINTERVENTION = "no_reintervention"


@dataclass(frozen=True)
class Adjudication:
    first: Decision
    second: Decision
    arbiter: Decision | None = None
    concordant: bool = True

    def __post_init__(self) -> None:
        agree = self.first is self.second
        if agree and self.arbiter is not None:
            raise ValueError("a concordant read has no arbiter")
        if not agree and self.arbiter is None:
            raise ValueError("a discordant read needs an arbiter")

    @property
    def decision(self) -> Decision:
        if self.first is self.second:
            return self.first
        if self.arbiter is None:
            raise ValueError("a discordant read has no arbiter")
        return self.arbiter


def adjudicate(
    first: Decision,
    second: Decision,
    arbiter: Decision | None = None,
) -> Adjudication:
    """Record a blind double read, admitting the arbiter only when it is needed."""
    if first is second:
        return Adjudication(first=first, second=second, concordant=True)
    if arbiter is None:
        raise ValueError("disagreement requires a third adjudicator")
    return Adjudication(first=first, second=second, arbiter=arbiter, concordant=False)


def outcome_from(adjudication: Adjudication, hours: float | None, escalated: bool) -> Outcome:
    """Turn an adjudicated decision into the endpoint object the label reads."""
    from astra_triage.endpoints.definition import OutcomeKind

    if adjudication.decision is Decision.REINTERVENTION:
        return Outcome(kind=OutcomeKind.REINTERVENTION, reintervention_hours=hours)
    if escalated:
        return Outcome(
            kind=OutcomeKind.ANTIBIOTIC_ESCALATION,
            antibiotic_escalated=True,
            reintervention_hours=None,
        )
    return Outcome(kind=OutcomeKind.UNEVENTFUL, reintervention_hours=None)
