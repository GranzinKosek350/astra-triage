"""Late-fusion transformer, the unconstrained baseline of Table 2.

Ref: Table 2, row "Late-fusion Transformer (unconstrained)"; Sec. 3.2, where the
representation that reads the sampling levels is argued to bound the one that does
not.

Each stream gets its own transformer encoder and is summarised independently; the
two summaries are then joined with the static vector. The arm is unconstrained in
the precise sense the article uses: the observation counts and their ratios are
supplied as features, so the arm may read sampling intensity directly. That is the
handicap it is given, and it is the reason the comparison is reported as parity
rather than as a win.
"""

from __future__ import annotations

import torch
from torch import nn

from astra_triage.comparators.base import ArmResult
from astra_triage.model.config import AstraConfig
from astra_triage.model.head import StaticEncoder
from astra_triage.model.time_encoding import TimeEncoder
from astra_triage.streams.case import CaseBatch, Episode, build_batch


class _StreamStack(nn.Module):
    def __init__(self, width: int, config: AstraConfig) -> None:
        super().__init__()
        self.values: nn.Sequential = nn.Sequential(
            nn.Linear(width, config.encoder_hidden),
            nn.GELU(),
            nn.Linear(config.encoder_hidden, config.d_model),
        )
        self.time: TimeEncoder = TimeEncoder(config)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.heads,
            dim_feedforward=2 * config.d_model,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder: nn.TransformerEncoder = nn.TransformerEncoder(layer, num_layers=2)

    def forward(
        self,
        values: torch.Tensor,
        tau: torch.Tensor,
        gap: torch.Tensor,
        mask: torch.Tensor,
        index: torch.Tensor,
    ) -> torch.Tensor:
        hidden: torch.Tensor = self.values(values) + self.time(tau, gap, index)
        encoded: torch.Tensor = self.encoder(hidden, src_key_padding_mask=~mask)
        weights = mask.to(encoded.dtype).unsqueeze(-1)
        pooled = (encoded * weights).sum(dim=1) / torch.clamp(weights.sum(dim=1), min=1.0)
        return pooled


class _LateFusionNet(nn.Module):
    def __init__(self, config: AstraConfig) -> None:
        super().__init__()
        self.ultrasound: _StreamStack = _StreamStack(config.ultrasound_width, config)
        self.markers: _StreamStack = _StreamStack(config.marker_width, config)
        self.static: StaticEncoder = StaticEncoder(config)
        count_width = 5
        self.head: nn.Sequential = nn.Sequential(
            nn.LayerNorm(3 * config.d_model + count_width),
            nn.Linear(3 * config.d_model + count_width, config.head_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.head_hidden, 1),
        )

    def forward(self, batch: CaseBatch) -> torch.Tensor:
        index = batch.index_hours
        ultrasound = self.ultrasound(
            batch.ultrasound.values,
            batch.ultrasound.tau_hours,
            batch.ultrasound.gap_hours,
            batch.ultrasound.mask,
            index,
        )
        markers = self.markers(
            batch.markers.values,
            batch.markers.tau_hours,
            batch.markers.gap_hours,
            batch.markers.mask,
            index,
        )
        counts = self._counts(batch)
        fused = torch.cat([ultrasound, markers, self.static(batch.static), counts], dim=-1)
        logits: torch.Tensor = self.head(fused).squeeze(-1)
        return logits

    @staticmethod
    def _counts(batch: CaseBatch) -> torch.Tensor:
        u = batch.ultrasound.mask.sum(dim=1).to(torch.float32)
        m = batch.markers.mask.sum(dim=1).to(torch.float32)
        total = torch.clamp(u + m, min=1.0)
        return torch.stack([u / 10.0, m / 10.0, (u + m) / 20.0, torch.log1p(u + m) / 3.0, u / total], dim=-1)


class LateFusionTransformer(nn.Module):
    name: str = "late_fusion_transformer"
    learned: bool = True

    def __init__(self, config: AstraConfig | None = None) -> None:
        super().__init__()
        self.net: _LateFusionNet = _LateFusionNet(config if config is not None else AstraConfig())
        self.capacity: int = 32

    def forward(self, batch: CaseBatch) -> torch.Tensor:
        logits: torch.Tensor = self.net(batch)
        return logits

    def score(self, episodes: list[Episode]) -> ArmResult:
        self.eval()
        with torch.no_grad():
            batch = build_batch(episodes, capacity=self.capacity)
            probability = torch.sigmoid(self.forward(batch))
        return ArmResult(name=self.name, scores=probability)
