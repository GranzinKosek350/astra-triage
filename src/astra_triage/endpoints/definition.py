"""Primary endpoint and index-time contract.

Ref: Sec. 2.1, "Endpoint and adjudication", and Eq. (1).

The endpoint is a re-intervention, surgical or percutaneous, for an
intra-abdominal infection inside a fixed window measured from the index time.
The index time is the moment the triage question is asked, so an observation
enters the model only when its timestamp is at or before the index time. The
window is closed at both ends: a re-intervention on day 7 counts, day 8 does not,
because the ward still holds the triage decision inside the first week and does
not hold it afterwards.

The complement is not "no event". It is the treated-and-improved course: the
antibiotic plan is escalated and the child recovers without an operation. That
distinction makes the label an action rather than a diagnosis, which is what the
score is asked to predict.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

HOURS_PER_DAY: float = 24.0
PRIMARY_WINDOW_DAYS: float = 7.0
PRIMARY_WINDOW_HOURS: float = PRIMARY_WINDOW_DAYS * HOURS_PER_DAY

# The score is read at the pre-specified operating point throughout the results,
# so the threshold belongs to the endpoint definition, not to a tuning loop.
TRIAGE_THRESHOLD: float = 0.20

# Observation age beyond which a stream is treated as unknown rather than stable.
STALE_AFTER_HOURS: float = 24.0


class OutcomeKind(StrEnum):
    REINTERVENTION = "reintervention"
    ANTIBIOTIC_ESCALATION = "antibiotic_escalation"
    UNEVENTFUL = "uneventful"


class TriageBand(StrEnum):
    LOW = "low"
    BORDERLINE = "borderline"
    HIGH = "high"


@dataclass(frozen=True)
class Outcome:
    """What happened after the index time.

    ``reintervention_hours`` is measured from the index time and is ``None`` when
    no re-intervention took place before discharge or before the end of follow-up.
    """

    kind: OutcomeKind
    reintervention_hours: float | None = None
    antibiotic_escalated: bool = False
    follow_up_hours: float = PRIMARY_WINDOW_HOURS

    def __post_init__(self) -> None:
        if self.reintervention_hours is not None and self.reintervention_hours < 0:
            raise ValueError("a re-intervention cannot precede the index time")
        if self.follow_up_hours < 0:
            raise ValueError("follow-up cannot be negative")
        if self.kind is OutcomeKind.REINTERVENTION and self.reintervention_hours is None:
            raise ValueError("a re-intervention outcome needs its time")
        if self.kind is not OutcomeKind.REINTERVENTION and self.reintervention_hours is not None:
            raise ValueError("a non-surgical outcome cannot carry a re-intervention time")


def label(outcome: Outcome, window_hours: float = PRIMARY_WINDOW_HOURS) -> int:
    """Positive when a re-intervention falls inside the closed window.

    An event recorded after the window closes is a non-event for triage: the ward
    no longer faces the decision the score is meant to support.
    """
    if outcome.kind is not OutcomeKind.REINTERVENTION:
        return 0
    recorded = outcome.reintervention_hours
    if recorded is None:
        return 0
    return int(0.0 <= recorded <= window_hours)


def index_visible(observation_hours: float, index_hours: float) -> bool:
    """Eq. (1): only observations stamped at or before the index time are usable."""
    return observation_hours <= index_hours


def band(probability: float) -> TriageBand:
    """Three-band read-out used by the case table; 0.20 and 0.50 are the cut points."""
    if probability < TRIAGE_THRESHOLD:
        return TriageBand.LOW
    if probability < 0.50:
        return TriageBand.BORDERLINE
    return TriageBand.HIGH
