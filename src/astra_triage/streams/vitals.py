"""Bedside vital signs and the clinical items the two named rules need.

Ref: Table 2, rows "Paediatric early-warning score" and "Paediatric Appendicitis
Score".

Neither instrument is one of the two model streams, but both appear in the
comparison table, so the ward record has to hold what they read. They are optional
on an episode: when the block is absent the rule cannot be evaluated and says so
rather than imputing a normal value.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Vitals:
    heart_rate: float
    respiratory_rate: float
    systolic_pressure: float
    temperature_c: float
    oxygen_saturation: float
    consciousness: str = "alert"

    def __post_init__(self) -> None:
        if self.consciousness not in {"alert", "voice", "pain", "unresponsive"}:
            raise ValueError("the consciousness item uses the four-level scale")
        if self.heart_rate <= 0 or self.respiratory_rate <= 0 or self.systolic_pressure <= 0:
            raise ValueError("a vital sign of zero or below is a recording error")
        if not 20.0 <= self.temperature_c <= 45.0:
            raise ValueError("temperature outside any physiological range")
        if not 0.0 <= self.oxygen_saturation <= 100.0:
            raise ValueError("saturation is a percentage")


@dataclass(frozen=True)
class ClinicalItems:
    """The bedside items of the appendicitis score, plus the abdominal examination."""

    migration_to_right_lower_quadrant: bool = False
    anorexia: bool = False
    nausea_or_vomiting: bool = False
    tenderness_right_lower_quadrant: bool = False
    rebound_or_cough_tenderness: bool = False
    temperature_above_38: bool = False
