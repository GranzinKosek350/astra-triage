"""Patient-level and temporal partitioning.

Ref: Table 1, "Derivation (5 sites, 3 regions)", "Temporal (1 site)", and Sec. 3.1
on the accrual window.

A patient contributes more than one episode in this cohort, so a split by episode
would leak a child across folds. The partition is checked at patient level and fails
loudly. The temporal split is checked by calendar position as well: the temporal
episodes must not precede the derivation window.
"""

from __future__ import annotations

from collections import defaultdict

from astra_triage.cohort.errors import RecordError
from astra_triage.streams.case import Episode

SPLITS: tuple[str, ...] = ("train", "validation", "test", "temporal", "external", "prospective")


def select(episodes: list[Episode], split: str) -> list[Episode]:
    rows = [e for e in episodes if e.split == split]
    if not rows:
        raise RecordError(f"no episode carries split={split}")
    return rows


def patient_overlap(episodes: list[Episode]) -> dict[str, list[str]]:
    """Patients appearing in more than one split, with the splits they touch."""
    found: dict[str, set[str]] = defaultdict(set)
    for episode in episodes:
        found[episode.patient_id].add(episode.split)
    return {patient: sorted(splits) for patient, splits in found.items() if len(splits) > 1}


def assert_patient_disjoint(episodes: list[Episode]) -> None:
    overlap = patient_overlap(episodes)
    if overlap:
        example = next(iter(overlap.items()))
        raise RecordError(f"{len(overlap)} patients straddle two splits, starting with {example[0]} in {example[1]}")


def site_distribution(episodes: list[Episode]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for episode in episodes:
        counts[episode.site] += 1
    return dict(sorted(counts.items()))


def event_count(episodes: list[Episode]) -> tuple[int, float]:
    events = sum(e.label for e in episodes)
    return events, 100.0 * events / len(episodes)
