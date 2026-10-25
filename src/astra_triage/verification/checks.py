"""The check record shared by both verification layers.

A status is one of four words and nothing else. ``PASS`` means the check ran and the
quantity it names was reproduced inside its tolerance. ``FAIL`` means it ran and was
not. ``BLOCKED`` means it needs records this release does not hold. ``NOT_RUN`` means
the check was not executed, including the case where it would have been trivially true.
No check may report a number it did not compute.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

FAMILIES: tuple[str, ...] = ("code", "manuscript", "record", "conformance", "hygiene")
STATUSES: tuple[str, ...] = ("PASS", "FAIL", "NOT_RUN", "BLOCKED")


@dataclass(frozen=True)
class Check:
    identifier: str
    family: str
    statement: str
    location: str
    status: str
    observed: Any = None
    expected: Any = None
    tolerance: float | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    note: str = ""

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"unknown check family: {self.family}")
        if self.status not in STATUSES:
            raise ValueError(f"unknown status: {self.status}")
        if self.status == "PASS" and self.observed is None:
            raise ValueError("a passing check carries what it observed")

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.identifier,
            "family": self.family,
            "statement": self.statement,
            "location": self.location,
            "status": self.status,
        }
        if self.observed is not None:
            payload["observed"] = self.observed
        if self.expected is not None:
            payload["expected"] = self.expected
        if self.tolerance is not None:
            payload["tolerance"] = self.tolerance
        if self.detail:
            payload["detail"] = self.detail
        if self.note:
            payload["note"] = self.note
        return payload


def verdict(observed: float, expected: float, tolerance: float) -> str:
    return "PASS" if abs(observed - expected) <= tolerance else "FAIL"


def counts(checks: list[Check]) -> dict[str, int]:
    summary = dict.fromkeys(STATUSES, 0)
    for check in checks:
        summary[check.status] += 1
    return summary


def overall(checks: list[Check]) -> str:
    """The release-level verdict: any failed check blocks a clean verification."""
    if not checks:
        return "UNVERIFIED"
    if any(check.status == "FAIL" for check in checks):
        return "PARTIALLY_VERIFIED"
    if any(check.status in {"BLOCKED", "NOT_RUN"} for check in checks):
        return "PARTIALLY_VERIFIED"
    return "VERIFIED"
