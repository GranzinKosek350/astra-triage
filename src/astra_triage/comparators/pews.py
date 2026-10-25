"""Paediatric early-warning score, the bedside comparator of Table 2.

Ref: Table 2, row "Paediatric early-warning score"; Sec. 3.2, "early warning score
for children of 12.5".

The instrument totals six items on an age-banded scale: heart rate, respiratory
rate and systolic pressure scored against the band for the child's age, and
saturation, temperature and consciousness scored absolutely. The article does not
print the band table, so the bands below are the ordinary ward bands for the four
paediatric age groups and are recorded as an implementation choice. The total is
mapped to a probability by the fitted link.

An episode without a vital-sign block has no PEWS. The arm reports itself
unavailable rather than substituting a normal value, which would silently deflate
the score.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from astra_triage.comparators.base import LinkedRule, RuleUnavailable
from astra_triage.streams.case import Episode
from astra_triage.streams.vitals import Vitals

AGE_BANDS: tuple[tuple[float, float, float, float, float], ...] = (
    (0.0, 1.0, 100.0, 160.0, 30.0),
    (1.0, 5.0, 90.0, 140.0, 24.0),
    (6.0, 12.0, 70.0, 120.0, 18.0),
    (13.0, 18.0, 60.0, 100.0, 12.0),
)
RESPIRATORY_UPPER: tuple[float, ...] = (50.0, 40.0, 30.0, 20.0)
SYSTOLIC_LOWER: tuple[float, ...] = (70.0, 80.0, 90.0, 100.0)


@dataclass(frozen=True)
class PewsTotal:
    points: int
    items: dict[str, int]


def _band_index(age: float) -> int:
    for index, entry in enumerate(AGE_BANDS):
        if entry[0] <= age < entry[1]:
            return index
    return len(AGE_BANDS) - 1


def _deviation_points(value: float, low: float, high: float) -> int:
    span = high - low
    if low <= value <= high:
        return 0
    distance = (low - value) if value < low else (value - high)
    fraction = distance / span
    if fraction <= 0.15:
        return 1
    if fraction <= 0.30:
        return 2
    return 3


def score_vitals(vitals: Vitals, age_years: float) -> PewsTotal:
    index = _band_index(age_years)
    _, _, heart_low, heart_high, respiratory_low = AGE_BANDS[index]
    items = {
        "heart_rate": _deviation_points(vitals.heart_rate, heart_low, heart_high),
        "respiratory_rate": _deviation_points(vitals.respiratory_rate, respiratory_low, RESPIRATORY_UPPER[index]),
        "systolic_pressure": _deviation_points(
            vitals.systolic_pressure, SYSTOLIC_LOWER[index], SYSTOLIC_LOWER[index] + 40.0
        ),
        "saturation": _saturation_points(vitals.oxygen_saturation),
        "temperature": _temperature_points(vitals.temperature_c),
        "consciousness": {"alert": 0, "voice": 1, "pain": 2, "unresponsive": 3}[vitals.consciousness],
    }
    return PewsTotal(points=sum(items.values()), items=items)


def _saturation_points(saturation: float) -> int:
    if saturation >= 96.0:
        return 0
    if saturation >= 94.0:
        return 1
    if saturation >= 92.0:
        return 2
    return 3


def _temperature_points(temperature: float) -> int:
    if 36.0 <= temperature < 38.0:
        return 0
    if temperature < 36.0 or temperature >= 40.0:
        return 3
    if temperature >= 39.0:
        return 2
    return 1


class PaediatricEarlyWarning(LinkedRule):
    name: str = "paediatric_early_warning_score"

    def totals(self, episodes: list[Episode]) -> torch.Tensor:
        points = []
        for episode in episodes:
            if episode.vitals is None:
                raise RuleUnavailable(f"episode {episode.episode_id} carries no vital-sign block")
            points.append(float(score_vitals(episode.vitals, episode.age_years).points))
        return torch.tensor(points, dtype=torch.float32)
