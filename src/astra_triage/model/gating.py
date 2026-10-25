"""Gated pooling over recency, and the cross-stream reliability gate.

Ref: Sec. 2.3.0.2, "a gated pooling layer to assign weights to both the recency of
observations and the reliability of the streams used for the observations".

Two gates. The first weights the readings of one stream against each other: a
learned score per reading plus a coefficient on the log decay of its age, normalised
over the live readings. The second mixes the two stream summaries dimension by
dimension, so the model can lean on the laboratory branch where the images are
uninformative and on the images where the markers are flat.

Both are blind to how many readings a stream contains. A duplicated reading splits
its own weight and leaves the pooled vector unchanged, which is the property the
admissible-set audit measures.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from astra_triage.model.config import AstraConfig
from astra_triage.model.ops import masked_softmax


class RecencyPooling(nn.Module):
    def __init__(self, config: AstraConfig) -> None:
        super().__init__()
        self.decay_hours: float = config.decay_hours
        self.score: nn.Linear = nn.Linear(config.d_model, 1)
        self.decay_coefficient: nn.Parameter = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        embeddings: torch.Tensor,
        mask: torch.Tensor,
        tau_hours: torch.Tensor,
        index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        age = torch.clamp(index.unsqueeze(-1) - tau_hours, min=0.0)
        base = self.score(embeddings).squeeze(-1)
        recency = -age / self.decay_hours
        weights = masked_softmax(base + F.softplus(self.decay_coefficient) * recency, mask)
        pooled = (weights.unsqueeze(-1) * embeddings).sum(dim=1)
        return pooled, weights


class StreamGate(nn.Module):
    def __init__(self, config: AstraConfig) -> None:
        super().__init__()
        self.mix: nn.Sequential = nn.Sequential(
            nn.Linear(2 * config.d_model, config.gate_hidden),
            nn.GELU(),
            nn.Linear(config.gate_hidden, config.d_model),
        )

    def forward(self, ultrasound: torch.Tensor, markers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate = torch.sigmoid(self.mix(torch.cat([ultrasound, markers], dim=-1)))
        fused = gate * ultrasound + (1.0 - gate) * markers
        return fused, gate.mean(dim=-1)
