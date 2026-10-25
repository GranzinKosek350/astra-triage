"""Per-site discrimination and leave-one-site-out refitting.

Ref: Table 4, panel A and panel B, and Sec. 3.4.

Panel A reads each site's own episodes from the same fitted model. Panel B refits on
the four remaining sites and scores the fifth, which is the only transport statement
the internal cohort can support. The heterogeneity figure is computed from the same
panel-A areas, so the two cannot drift apart.

A site with one class present is left out of the panel rather than given a degenerate
area, and a panel with fewer than two scored sites cannot be pooled.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from astra_triage.metrics.calibration import expected_calibration_error
from astra_triage.metrics.discrimination import auroc
from astra_triage.metrics.heterogeneity import from_areas
from astra_triage.streams.case import Episode
from astra_triage.training.loop import train_arm
from astra_triage.training.optim import TrainConfig


@dataclass(frozen=True)
class SiteRow:
    site: str
    episodes: int
    events: int
    scores: dict[str, float]


def per_site(arms: dict[str, torch.Tensor], episodes: list[Episode]) -> list[SiteRow]:
    labels = torch.tensor([float(e.label) for e in episodes], dtype=torch.float32)
    sites = sorted({e.site for e in episodes})
    rows: list[SiteRow] = []
    for site in sites:
        mask = torch.tensor([e.site == site for e in episodes], dtype=torch.bool)
        if int(mask.sum()) == 0:
            continue
        events = int(labels[mask].sum())
        if events == 0 or events == int(mask.sum()):
            continue
        values = {name: auroc(scores[mask], labels[mask]) for name, scores in arms.items()}
        rows.append(
            SiteRow(
                site=site,
                episodes=int(mask.sum()),
                events=int(labels[mask].sum()),
                scores=values,
            )
        )
    return rows


def heterogeneity_by_arm(arms: dict[str, torch.Tensor], episodes: list[Episode]) -> dict[str, dict[str, float]]:
    rows = per_site(arms, episodes)
    if len(rows) < 2:
        raise ValueError("pooling between sites needs at least two scored sites")
    result: dict[str, dict[str, float]] = {}
    for name in arms:
        areas = [row.scores[name] for row in rows]
        positives = [row.events for row in rows]
        negatives = [row.episodes - row.events for row in rows]
        result[name] = from_areas(areas, positives, negatives)
    return result


@dataclass(frozen=True)
class LeaveOutRow:
    site: str
    auroc: float
    episodes: int
    events: int
    ece: float


def leave_one_site_out(
    factory: Callable[[], nn.Module],
    episodes: list[Episode],
    config: TrainConfig,
    device: torch.device,
    seed: int,
) -> list[LeaveOutRow]:
    sites = sorted({e.site for e in episodes})
    rows: list[LeaveOutRow] = []
    for site in sites:
        training = [e for e in episodes if e.site != site]
        held = [e for e in episodes if e.site == site]
        if not training or not held:
            continue
        arm = factory()
        train_arm(arm, training, config, device, seed=seed)
        arm.eval()
        from astra_triage.training.loop import predict

        with torch.no_grad():
            scores = predict(arm, held)
        labels = torch.tensor([float(e.label) for e in held], dtype=torch.float32)
        rows.append(
            LeaveOutRow(
                site=site,
                auroc=auroc(scores, labels),
                episodes=len(held),
                events=int(labels.sum()),
                ece=expected_calibration_error(scores, labels),
            )
        )
    return rows
