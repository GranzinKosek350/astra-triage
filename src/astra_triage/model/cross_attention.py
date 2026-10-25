"""Cross-stream attention, Eq. (2).

Ref: Sec. 2.3.0.2, Eq. (2).

    Z = Concat(MHA(E_u, E_m, E_m), MHA(E_m, E_u, E_u))

Each ultrasound reading queries the marker sequence and each marker reading queries
the ultrasound sequence, so a free-fluid reading on a morning when C-reactive
protein peaks is not the same object as the same reading on a falling curve. The
two branches are returned separately because the gated pooling of Sec. 2.3.0.2 is
applied to each branch before the concatenation is read as one vector.

When a stream has no reading at all, the encoder has already replaced it with a single
structural-absence slot, so that side of the concatenation is one learned token
attending over the other sequence. This is the only way Eq. (2) can be evaluated for a
cohort that records one stream and not the other, and it is why the mask reaching this
module always has at least one live entry.
"""

from __future__ import annotations

import torch
from torch import nn

from astra_triage.model.config import AstraConfig


class CrossStreamAttention(nn.Module):
    def __init__(self, config: AstraConfig) -> None:
        super().__init__()
        self.enabled: bool = config.use_cross_attention
        self.ultrasound_reads_markers: nn.MultiheadAttention = nn.MultiheadAttention(
            config.d_model, config.heads, dropout=config.dropout, batch_first=True
        )
        self.markers_read_ultrasound: nn.MultiheadAttention = nn.MultiheadAttention(
            config.d_model, config.heads, dropout=config.dropout, batch_first=True
        )
        self.norm_ultrasound: nn.LayerNorm = nn.LayerNorm(config.d_model)
        self.norm_markers: nn.LayerNorm = nn.LayerNorm(config.d_model)

    def _attend(
        self,
        module: nn.MultiheadAttention,
        query: torch.Tensor,
        query_mask: torch.Tensor,
        source: torch.Tensor,
        source_mask: torch.Tensor,
        norm: nn.LayerNorm,
    ) -> torch.Tensor:
        attended, _ = module(
            query,
            source,
            source,
            key_padding_mask=~source_mask,
            need_weights=False,
        )
        attended_tensor: torch.Tensor = attended
        residual: torch.Tensor = norm(query + attended_tensor)
        return residual * query_mask.unsqueeze(-1).to(residual.dtype)

    def forward(
        self,
        ultrasound: torch.Tensor,
        ultrasound_mask: torch.Tensor,
        markers: torch.Tensor,
        marker_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.enabled:
            return ultrasound, markers

        from_markers = self._attend(
            self.ultrasound_reads_markers,
            ultrasound,
            ultrasound_mask,
            markers,
            marker_mask,
            self.norm_ultrasound,
        )
        from_ultrasound = self._attend(
            self.markers_read_ultrasound,
            markers,
            marker_mask,
            ultrasound,
            ultrasound_mask,
            self.norm_markers,
        )
        return from_markers, from_ultrasound
