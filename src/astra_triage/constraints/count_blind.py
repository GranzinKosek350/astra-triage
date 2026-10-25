"""The count-blind constraint, measured on the estimator itself.

Ref: Sec. 2.3.0.1, "The estimator must also be count-blind: the number of
observations and any of its monotonic function must not be present in the model.
The expression must also be roughly unchanged by the position".

Two probes are used, and neither reads the estimator's internals.

Replication. Every reading of one stream is repeated k times at the same timestamp
and with the same values. A count-blind estimator returns the same probability,
because duplicating a reading adds no information. An estimator that counts, sums
or takes a maximum over readings does not.

Permutation. The readings of each stream are reordered. A position-dependent
estimator changes its output; a set-based one does not.

Both are exact statements about the architecture, so the tolerances below are
numerical rather than clinical.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import nn

from astra_triage.model.astra import Astra
from astra_triage.streams.case import Episode, build_batch

REPLICATION_TOLERANCE: float = 1e-5
PERMUTATION_TOLERANCE: float = 1e-5


def replicate(episodes: list[Episode], factor: int = 3) -> list[Episode]:
    if factor < 1:
        raise ValueError("a replication factor is at least one")
    rows = []
    for episode in episodes:
        ultrasound = tuple(o for o in episode.ultrasound for _ in range(factor))
        markers = tuple(o for o in episode.markers for _ in range(factor))
        rows.append(replace(episode, ultrasound=ultrasound, markers=markers))
    return rows


def permute(episodes: list[Episode], seed: int = 7) -> list[Episode]:
    rng = torch.Generator().manual_seed(seed)
    rows = []
    for episode in episodes:
        ultrasound = list(episode.ultrasound)
        markers = list(episode.markers)
        if len(ultrasound) > 1:
            order = torch.randperm(len(ultrasound), generator=rng).tolist()
            ultrasound = [ultrasound[i] for i in order]
        if len(markers) > 1:
            order = torch.randperm(len(markers), generator=rng).tolist()
            markers = [markers[i] for i in order]
        rows.append(replace(episode, ultrasound=tuple(ultrasound), markers=tuple(markers)))
    return rows


def _probabilities(arm: nn.Module, episodes: list[Episode], capacity: int) -> torch.Tensor:
    arm.eval()
    with torch.no_grad():
        output = arm(build_batch(episodes, capacity=capacity))
        probability: torch.Tensor = output.probability if isinstance(arm, Astra) else torch.sigmoid(output)
    return probability


def replication_deviation(arm: nn.Module, episodes: list[Episode], factor: int = 3) -> float:
    capacity = 32 * factor
    base = _probabilities(arm, episodes, capacity)
    doubled = _probabilities(arm, replicate(episodes, factor), capacity)
    return float((base - doubled).abs().max())


def permutation_deviation(arm: nn.Module, episodes: list[Episode], seed: int = 7) -> float:
    base = _probabilities(arm, episodes, 32)
    shuffled = _probabilities(arm, permute(episodes, seed), 32)
    return float((base - shuffled).abs().max())


@dataclass(frozen=True)
class CountBlindResult:
    replication: float
    permutation: float
    tolerance: float

    @property
    def passes(self) -> bool:
        return self.replication <= self.tolerance and self.permutation <= self.tolerance


def audit(
    arm: nn.Module,
    episodes: list[Episode],
    factor: int = 3,
    tolerance: float = REPLICATION_TOLERANCE,
) -> CountBlindResult:
    return CountBlindResult(
        replication=replication_deviation(arm, episodes, factor),
        permutation=permutation_deviation(arm, episodes),
        tolerance=tolerance,
    )
