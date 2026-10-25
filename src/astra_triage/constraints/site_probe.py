"""The site probe: does the merged representation still know where it came from?

Ref: Sec. 2.3.0.1, "The expression must also be roughly unchanged by the position,
whereby the change in it is detected using the site-classification instead of just
stating it"; Sec. 2.3.0.3 on the gradient-reversal term.

The probe is a fresh classifier trained on frozen representations, so it measures
what the representation carries rather than what the model's own adversarial head has
learned to suppress. It is fitted on one part of the cohort and scored on another;
training and scoring on the same cases would report memorisation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from astra_triage.model.astra import Astra
from astra_triage.model.config import AstraConfig
from astra_triage.model.head import SiteHead
from astra_triage.streams.case import Episode, build_batch


@dataclass(frozen=True)
class ProbeResult:
    accuracy: float
    majority_accuracy: float
    episodes: int
    sites: int

    @property
    def lift(self) -> float:
        return self.accuracy - self.majority_accuracy


def representations(arm: Astra, episodes: list[Episode]) -> tuple[torch.Tensor, torch.Tensor]:
    arm.eval()
    with torch.no_grad():
        output = arm(build_batch(episodes, capacity=arm.config.capacity))
    return output.representation.detach(), batch_sites(episodes)


def batch_sites(episodes: list[Episode]) -> torch.Tensor:
    order = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    return torch.tensor([order[e.site] for e in episodes], dtype=torch.long)


def probe(
    arm: Astra,
    fit_episodes: list[Episode],
    score_episodes: list[Episode],
    config: AstraConfig | None = None,
    epochs: int = 200,
    learning_rate: float = 1e-2,
    seed: int = 11,
) -> ProbeResult:
    if not fit_episodes or not score_episodes:
        raise ValueError("the probe needs two disjoint sets")
    settings = config if config is not None else arm.config
    torch.manual_seed(seed)
    head = SiteHead(settings)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=1e-3)
    features, sites = representations(arm, fit_episodes)
    held_features, held_sites = representations(arm, score_episodes)
    head.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = head(features)
        loss = nn.functional.cross_entropy(logits, sites)
        torch.autograd.backward(loss)
        optimizer.step()
    head.eval()
    with torch.no_grad():
        predicted = head(held_features).argmax(dim=-1)
    accuracy = float((predicted == held_sites).to(torch.float32).mean())
    counts = torch.bincount(held_sites, minlength=settings.site_count).to(torch.float32)
    majority = float(counts.max().item() / max(float(counts.sum().item()), 1.0))
    return ProbeResult(
        accuracy=accuracy,
        majority_accuracy=majority,
        episodes=len(score_episodes),
        sites=int(counts.gt(0).sum()),
    )
