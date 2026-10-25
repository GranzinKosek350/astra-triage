"""C-reactive protein threshold rule.

Ref: Table 2, row "C-reactive protein threshold rule"; abstract, "net benefit at
the 0.20 threshold was 0.060 per episode against 0.033 for a C-reactive protein
rule".

The rule reads the last C-reactive protein before the index time and nothing else.
It is deliberately the crudest arm in the table, and it is the one the decision
curve compares against, so it must be exactly the single-marker read and not a
marker model with a different name.
"""

from __future__ import annotations

import torch

from astra_triage.comparators.base import LinkedRule, RuleUnavailable
from astra_triage.streams.case import Episode

CRP_INDEX: int = 4


def last_crp(episode: Episode) -> float:
    visible = episode.markers
    if not visible:
        raise RuleUnavailable(f"episode {episode.episode_id} carries no marker panel before the index time")
    newest = max(visible, key=lambda observation: observation.tau_hours)
    return newest.values[CRP_INDEX]


class CReactiveProteinRule(LinkedRule):
    name: str = "crp_threshold_rule"

    def totals(self, episodes: list[Episode]) -> torch.Tensor:
        values = torch.tensor([last_crp(e) for e in episodes], dtype=torch.float32)
        return torch.log1p(values)
