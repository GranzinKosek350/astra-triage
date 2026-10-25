"""Paediatric appendicitis score, the second clinical rule of Table 2.

Ref: Table 2, row "Paediatric Appendicitis Score"; Sec. 3.2, "two medical
guidelines proposed by 15.9 and 16.7 by points".

Eight binary items totalling ten points: the six bedside items of the appendicitis
score, a white-cell count above 10 x 10^9/L, and a neutrophil fraction above 0.75,
both read from the last marker panel before the index time. The article does not
print the item weights, so the ordinary weights of the score are used. An episode
without the bedside block has no score, and the arm says so.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from astra_triage.comparators.base import LinkedRule, RuleUnavailable
from astra_triage.streams.case import Episode

LEUKOCYTOSIS_THRESHOLD: float = 10.0
NEUTROPHIL_FRACTION_THRESHOLD: float = 0.75


@dataclass(frozen=True)
class AppendicitisScore:
    points: int
    items: dict[str, int]


def _last_panel(episode: Episode) -> tuple[float, float] | None:
    visible = episode.markers
    if not visible:
        return None
    newest = max(visible, key=lambda observation: observation.tau_hours)
    wbc, neutrophil = newest.values[0], newest.values[1]
    return wbc, neutrophil


def score_episode(episode: Episode) -> AppendicitisScore:
    clinical = episode.clinical
    if clinical is None:
        raise RuleUnavailable(f"episode {episode.episode_id} carries no bedside clinical block")
    panel = _last_panel(episode)
    if panel is None:
        raise RuleUnavailable(f"episode {episode.episode_id} carries no bedside clinical block")
    wbc, neutrophil = panel
    fraction = neutrophil / wbc if wbc > 0 else 0.0
    items = {
        "migration": int(clinical.migration_to_right_lower_quadrant),
        "anorexia": int(clinical.anorexia),
        "nausea_or_vomiting": int(clinical.nausea_or_vomiting),
        "tenderness": 2 * int(clinical.tenderness_right_lower_quadrant),
        "rebound_or_cough": 2 * int(clinical.rebound_or_cough_tenderness),
        "temperature": int(clinical.temperature_above_38),
        "leukocytosis": 2 * int(wbc > LEUKOCYTOSIS_THRESHOLD),
        "neutrophil_shift": int(fraction > NEUTROPHIL_FRACTION_THRESHOLD),
    }
    return AppendicitisScore(points=sum(items.values()), items=items)


class PaediatricAppendicitisScore(LinkedRule):
    name: str = "paediatric_appendicitis_score"

    def totals(self, episodes: list[Episode]) -> torch.Tensor:
        return torch.tensor([float(score_episode(e).points) for e in episodes], dtype=torch.float32)
