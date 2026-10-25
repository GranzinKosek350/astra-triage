"""The fused triage representation.

Ref: Sec. 2.3.0.1 to Sec. 2.3.0.4, Eq. (1)-(3).

A case enters as two stamped sequences and a static vector, is filtered to the
index time by the stream layer, and leaves as a probability. The path is: encode
each reading with its descriptors and its two clocks, let each stream read the
other, pool by recency inside each stream, mix the two summaries by reliability,
concatenate the static embedding, and score with a two-layer head.

The learned objects that carry the argument of the article are all present here and
all switchable: the continuous-time encoding, the two attention directions, the
stale-stream token, and the site adapter with its adversarial head.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from astra_triage.comparators.base import ArmResult
from astra_triage.model.config import AstraConfig
from astra_triage.model.cross_attention import CrossStreamAttention
from astra_triage.model.encoders import StreamEncoder
from astra_triage.model.gating import RecencyPooling, StreamGate
from astra_triage.model.head import SiteHead, StaticEncoder, TriageHead
from astra_triage.objectives.site_loss import reverse_gradient
from astra_triage.streams.case import CaseBatch, Episode


@dataclass(frozen=True)
class AstraOutput:
    probability: torch.Tensor
    logits: torch.Tensor
    representation: torch.Tensor
    site_logits: torch.Tensor
    stream_gate: torch.Tensor
    ultrasound_weights: torch.Tensor
    marker_weights: torch.Tensor
    ultrasound_stale: torch.Tensor
    marker_stale: torch.Tensor


class Astra(nn.Module):
    reversal_strength: torch.Tensor

    def __init__(self, config: AstraConfig) -> None:
        super().__init__()
        self.config: AstraConfig = config
        self.ultrasound_encoder: StreamEncoder = StreamEncoder(config.ultrasound_width, config)
        self.marker_encoder: StreamEncoder = StreamEncoder(config.marker_width, config)
        self.fusion: CrossStreamAttention = CrossStreamAttention(config)
        self.ultrasound_pool: RecencyPooling = RecencyPooling(config)
        self.marker_pool: RecencyPooling = RecencyPooling(config)
        self.stream_gate: StreamGate = StreamGate(config)
        self.static_encoder: StaticEncoder = StaticEncoder(config)
        self.head: TriageHead = TriageHead(config)
        self.site_head: SiteHead = SiteHead(config)
        self.register_buffer("reversal_strength", torch.zeros((), dtype=torch.float32))

    def set_reversal_strength(self, strength: float) -> None:
        self.reversal_strength.fill_(float(strength))

    def forward(self, batch: CaseBatch) -> AstraOutput:
        index = batch.index_hours
        ultrasound, u_mask, u_stale = self.ultrasound_encoder(
            batch.ultrasound.values,
            batch.ultrasound.tau_hours,
            batch.ultrasound.gap_hours,
            batch.ultrasound.mask,
            index,
            batch.site,
        )
        markers, m_mask, m_stale = self.marker_encoder(
            batch.markers.values,
            batch.markers.tau_hours,
            batch.markers.gap_hours,
            batch.markers.mask,
            index,
            batch.site,
        )

        from_markers, from_ultrasound = self.fusion(ultrasound, u_mask, markers, m_mask)

        pooled_ultrasound, u_weights = self.ultrasound_pool(from_markers, u_mask, batch.ultrasound.tau_hours, index)
        pooled_markers, m_weights = self.marker_pool(from_ultrasound, m_mask, batch.markers.tau_hours, index)
        representation, gate = self.stream_gate(pooled_ultrasound, pooled_markers)
        static = self.static_encoder(batch.static)
        logits = self.head(torch.cat([representation, static], dim=-1))
        return AstraOutput(
            probability=torch.sigmoid(logits),
            logits=logits,
            representation=representation,
            site_logits=self.site_head(reverse_gradient(representation, float(self.reversal_strength.item()))),
            stream_gate=gate,
            ultrasound_weights=u_weights,
            marker_weights=m_weights,
            ultrasound_stale=u_stale,
            marker_stale=m_stale,
        )

    def score(self, episodes: list[Episode], capacity: int | None = None) -> ArmResult:
        """Score a set of episodes through the index-time path."""
        from astra_triage.streams.case import build_batch

        self.eval()
        with torch.no_grad():
            batch = build_batch(episodes, capacity=capacity or self.config.capacity)
            probability = self.forward(batch).probability
        return ArmResult(name="astra", scores=probability)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_astra(config: AstraConfig | None = None) -> Astra:
    return Astra(config if config is not None else AstraConfig())
