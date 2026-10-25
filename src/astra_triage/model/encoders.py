"""Per-stream encoder: descriptor vector, continuous time, site adapter, tokens.

Ref: Sec. 2.3.0.1 to Sec. 2.3.0.3.

An observation is encoded from its standardised descriptor vector and its two
clocks. When a stream carries no observation at or before the index time, one slot
is filled with the structural absence token and the remaining slots are masked, so
the rest of the network sees a well-formed sequence and cannot mistake absence for
a reading of zero.
"""

from __future__ import annotations

import torch
from torch import nn

from astra_triage.model.config import AstraConfig
from astra_triage.model.missingness import MissingnessTokens
from astra_triage.model.site_adapter import SiteAdapter
from astra_triage.model.time_encoding import TimeEncoder


class StreamEncoder(nn.Module):
    def __init__(self, value_width: int, config: AstraConfig) -> None:
        super().__init__()
        self.project: nn.Sequential = nn.Sequential(
            nn.Linear(value_width, config.encoder_hidden),
            nn.GELU(),
            nn.Linear(config.encoder_hidden, config.d_model),
            nn.LayerNorm(config.d_model),
        )
        self.time: TimeEncoder = TimeEncoder(config)
        self.tokens: MissingnessTokens = MissingnessTokens(config)
        self.adapter: SiteAdapter | None
        if config.use_site_adapter:
            self.adapter = SiteAdapter(config)
        else:
            self.adapter = None

    def forward(
        self,
        values: torch.Tensor,
        tau_hours: torch.Tensor,
        gap_hours: torch.Tensor,
        mask: torch.Tensor,
        index: torch.Tensor,
        site: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        width = int(values.shape[1])
        embeddings: torch.Tensor = self.project(values)
        if self.time.project is not None:
            embeddings = embeddings + self.time(tau_hours, gap_hours, index)

        present = mask.any(dim=-1)
        if not bool(present.all()):
            absent_rows = (~present).unsqueeze(-1).unsqueeze(-1)
            token = self.tokens.absent_embedding(embeddings, width)
            embeddings = torch.where(absent_rows, token, embeddings)
            mask = mask | (self.tokens.absent_flag(embeddings, width) & ~present.unsqueeze(-1))
            tau_hours = torch.where(present.unsqueeze(-1), tau_hours, torch.zeros_like(tau_hours))

        if self.adapter is not None:
            embeddings = self.adapter(embeddings, site)

        embeddings, stale = self.tokens(embeddings, mask, tau_hours, index, present)
        return embeddings, mask, stale
