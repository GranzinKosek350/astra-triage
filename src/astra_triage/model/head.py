"""Static embedding, two-layer head and the adversarial site head.

Ref: Sec. 2.3.0.2 (concatenation with the static embedding and a two-layer head),
Sec. 2.3.0.3 and Eq. (3) for the gradient-reversal term.

The site head reads the merged representation, not the input, so the adversarial
term constrains exactly what the article says it constrains: the fused vector must
not carry the site. The static vector is a separate path and keeps age and
procedure.
"""

from __future__ import annotations

import torch
from torch import nn

from astra_triage.model.config import AstraConfig


class StaticEncoder(nn.Module):
    def __init__(self, config: AstraConfig) -> None:
        super().__init__()
        self.project: nn.Sequential = nn.Sequential(
            nn.Linear(config.static_width, config.encoder_hidden),
            nn.GELU(),
            nn.Linear(config.encoder_hidden, config.d_model),
            nn.LayerNorm(config.d_model),
        )

    def forward(self, static: torch.Tensor) -> torch.Tensor:
        encoded: torch.Tensor = self.project(static)
        return encoded


class TriageHead(nn.Module):
    def __init__(self, config: AstraConfig) -> None:
        super().__init__()
        self.layers: nn.Sequential = nn.Sequential(
            nn.LayerNorm(2 * config.d_model),
            nn.Linear(2 * config.d_model, config.head_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.head_hidden, 1),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.layers(fused).squeeze(-1)
        return logits


class SiteHead(nn.Module):
    def __init__(self, config: AstraConfig) -> None:
        super().__init__()
        self.probe: nn.Sequential = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.gate_hidden),
            nn.GELU(),
            nn.Linear(config.gate_hidden, config.site_count),
        )

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.probe(representation)
        return logits
