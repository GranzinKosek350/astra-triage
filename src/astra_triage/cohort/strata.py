"""Pre-specified subgroups and the high-sparsity stratum.

Ref: Table 6 panel A, "Panel A. Pre-specified subgroups", and panel B, "Sparsity
strata (fraction of unobserved stream-days)".

The age groups are the four bands the table prints. The three clinical groups are
recorded backgrounds, not inferred from the procedure field: a child operated for
necrotising enterocolitis and a child operated for perforated appendicitis both
appear as laparotomies in the procedure field, so the background has to come from
the record. The high-sparsity group is the pooled unobserved share at or above
0.40, the lower edge of the high band in panel B.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from astra_triage.observation.sparsity import day_absence
from astra_triage.streams.case import Episode

HIGH_SPARSITY_EDGE: float = 0.40


@dataclass(frozen=True)
class Strata:
    name: str
    predicate: Callable[[Episode], bool]


def _age(episode: Episode, low: float, high: float) -> bool:
    return low <= episode.age_years < high


def sparsity_of(episode: Episode) -> float:
    return day_absence(episode).pooled_fraction


STRATA: tuple[Strata, ...] = (
    Strata("age_under_1", lambda e: _age(e, 0.0, 1.0)),
    Strata("age_1_to_5", lambda e: _age(e, 1.0, 6.0)),
    Strata("age_6_to_12", lambda e: _age(e, 6.0, 13.0)),
    Strata("age_13_to_17", lambda e: _age(e, 13.0, 18.0)),
    Strata("necrotising_enterocolitis", lambda e: "necrotising_enterocolitis" in e.backgrounds),
    Strata("complicated_appendicitis", lambda e: "complicated_appendicitis" in e.backgrounds),
    Strata("enterostomy", lambda e: "enterostomy" in e.backgrounds),
    Strata("high_sparsity", lambda e: sparsity_of(e) >= HIGH_SPARSITY_EDGE),
)


def members(episodes: list[Episode], name: str) -> list[Episode]:
    for entry in STRATA:
        if entry.name == name:
            return [e for e in episodes if entry.predicate(e)]
    raise KeyError(f"unknown subgroup: {name}")


def describe(episodes: list[Episode]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for entry in STRATA:
        selected = [e for e in episodes if entry.predicate(e)]
        rows.append(
            {
                "subgroup": entry.name,
                "episodes": float(len(selected)),
                "events": float(sum(e.label for e in selected)),
            }
        )
    return rows
