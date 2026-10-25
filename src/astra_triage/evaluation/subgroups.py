"""Subgroup and sparsity-stratum tables.

Ref: Table 6 panel A (pre-specified subgroups) and panel B (sparsity strata).

Panel A reads an arm inside each recorded subgroup. Panel B bands the cohort by the
pooled unobserved stream-day share and reports discrimination, calibration error and
slope in each band for two arms, which is the axis on which the constrained
representation is claimed to hold its calibration.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from astra_triage.cohort.strata import STRATA, sparsity_of
from astra_triage.metrics.calibration import calibration_slope, expected_calibration_error
from astra_triage.metrics.discrimination import auroc
from astra_triage.observation.sparsity import STRATUM_EDGES, STRATUM_NAMES, stratum
from astra_triage.streams.case import Episode


@dataclass(frozen=True)
class SubgroupRow:
    subgroup: str
    episodes: int
    events: int
    scores: dict[str, float]


def _labels(episodes: list[Episode]) -> torch.Tensor:
    return torch.tensor([float(e.label) for e in episodes], dtype=torch.float32)


def subgroup_table(arms: dict[str, torch.Tensor], episodes: list[Episode]) -> list[SubgroupRow]:
    rows: list[SubgroupRow] = []
    for entry in STRATA:
        mask = torch.tensor([entry.predicate(e) for e in episodes], dtype=torch.bool)
        count = int(mask.sum())
        labels = _labels(episodes)[mask] if count else torch.zeros(1)
        if count == 0 or int(labels.sum()) == 0 or int(labels.sum()) == count:
            continue
        rows.append(
            SubgroupRow(
                subgroup=entry.name,
                episodes=count,
                events=int(labels.sum()),
                scores={name: auroc(scores[mask], labels) for name, scores in arms.items()},
            )
        )
    return rows


@dataclass(frozen=True)
class StratumRow:
    stratum: str
    episodes: int
    events: int
    auroc: dict[str, float]
    ece: dict[str, float]
    slope: dict[str, float]


def sparsity_table(arms: dict[str, torch.Tensor], episodes: list[Episode]) -> list[StratumRow]:
    bounds = [0.0, *STRATUM_EDGES, 1.0]
    rows: list[StratumRow] = []
    fractions = torch.tensor([sparsity_of(e) for e in episodes], dtype=torch.float32)
    for index, name in enumerate(STRATUM_NAMES):
        low, high = bounds[index], bounds[index + 1]
        mask = (fractions >= low) & (fractions < high) if index < len(STRATUM_NAMES) - 1 else fractions >= low
        count = int(mask.sum())
        if count < 4:
            continue
        labels = _labels(episodes)[mask]
        if int(labels.sum()) == 0 or int(labels.sum()) == count:
            continue
        rows.append(
            StratumRow(
                stratum=name,
                episodes=count,
                events=int(labels.sum()),
                auroc={key: auroc(value[mask], labels) for key, value in arms.items()},
                ece={key: expected_calibration_error(value[mask], labels) for key, value in arms.items()},
                slope={key: calibration_slope(value[mask], labels)[1] for key, value in arms.items()},
            )
        )
    return rows


def assigned_stratum(episode: Episode) -> str:
    return stratum(sparsity_of(episode))
