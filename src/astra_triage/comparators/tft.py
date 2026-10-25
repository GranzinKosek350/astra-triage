"""Temporal Fusion Transformer, the published sequence baseline of Table 2.

Ref: Table 2, row "Temporal Fusion Transformer"; Sec. 3.2, "above those of Temporal
Fusion Transformer of 2.0 by percentage points".

The four blocks that make the published design are all present: a variable-selection
network that weights each input per step, a static covariate encoder that conditions
that selection, an LSTM encoder with interpretable multi-head attention over the
encoded steps, and gated residual blocks with skip connections around each stage.
The output is a single logit read at the index hour, which is the binary form of the
quantile head the forecasting design uses.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from astra_triage.comparators.base import ArmResult
from astra_triage.comparators.grid import grid_from_batch
from astra_triage.model.config import AstraConfig
from astra_triage.streams.case import CaseBatch, Episode, build_batch


class _GatedLinearUnit(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.left: nn.Linear = nn.Linear(width, width)
        self.right: nn.Linear = nn.Linear(width, width)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        gated: torch.Tensor = self.left(values) * torch.sigmoid(self.right(values))
        return gated


class _GatedResidualNetwork(nn.Module):
    def __init__(self, width: int, context: int | None, dropout: float) -> None:
        super().__init__()
        self.context: nn.Linear | None
        if context is None:
            self.context = None
        else:
            self.context = nn.Linear(context, width, bias=False)
        self.first: nn.Linear = nn.Linear(width, width)
        self.second: nn.Linear = nn.Linear(width, width)
        self.gate: _GatedLinearUnit = _GatedLinearUnit(width)
        self.norm: nn.LayerNorm = nn.LayerNorm(width)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.first(values)
        if context is not None and self.context is not None:
            hidden = hidden + self.context(context)
        hidden = F.elu(hidden)
        hidden = self.dropout(self.second(hidden))
        residual: torch.Tensor = self.norm(values + self.gate(hidden))
        return residual


class _VariableSelection(nn.Module):
    def __init__(self, inputs: int, width: int, dropout: float) -> None:
        super().__init__()
        self.project: nn.ModuleList = nn.ModuleList([nn.Linear(1, width) for _ in range(inputs)])
        self.per_variable: nn.ModuleList = nn.ModuleList(
            [_GatedResidualNetwork(width, width, dropout) for _ in range(inputs)]
        )
        self.weights: nn.Linear = nn.Linear(width, 1)

    def forward(self, values: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        transformed = torch.stack(
            [
                block(self.project[index](values[..., index : index + 1]), context)
                for index, block in enumerate(self.per_variable)
            ],
            dim=-2,
        )
        scores = self.weights(transformed).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        selected = (transformed * weights.unsqueeze(-1)).sum(dim=-2)
        return selected, weights


class _InterpretableAttention(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads: int = heads
        self.width: int = width
        self.query: nn.Linear = nn.Linear(width, width)
        self.key: nn.Linear = nn.Linear(width, width)
        self.value: nn.Linear = nn.Linear(width, width)
        self.out: nn.Linear = nn.Linear(width, width)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, steps, _ = values.shape
        head_width = self.width // self.heads
        query = self.query(values).view(batch, steps, self.heads, head_width).transpose(1, 2)
        key = self.key(values).view(batch, steps, self.heads, head_width).transpose(1, 2)
        value = self.value(values)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (head_width**0.5)
        scores = scores.masked_fill(~mask.unsqueeze(1).unsqueeze(-1), float("-inf"))
        weights = torch.softmax(scores, dim=-1).mean(dim=1)
        weights = self.dropout(weights)
        attended = torch.matmul(weights, value)
        projected: torch.Tensor = self.out(attended)
        return projected


class TFT(nn.Module):
    name: str = "temporal_fusion_transformer"
    learned: bool = True

    def __init__(self, config: AstraConfig | None = None) -> None:
        super().__init__()
        settings = config if config is not None else AstraConfig()
        self.capacity: int = settings.capacity
        width = settings.d_model
        self.width: int = width
        inputs = settings.ultrasound_width + settings.marker_width
        self.inputs: int = inputs
        self.static_variable: _VariableSelection = _VariableSelection(inputs + inputs, width, settings.dropout)
        self.static_to_width: nn.Linear = nn.Linear(settings.static_width, width)
        self.lstm: nn.LSTM = nn.LSTM(width, width, batch_first=True)
        self.enrichment: _GatedResidualNetwork = _GatedResidualNetwork(width, width, settings.dropout)
        self.attention: _InterpretableAttention = _InterpretableAttention(width, settings.heads, settings.dropout)
        self.attention_gate: _GatedLinearUnit = _GatedLinearUnit(width)
        self.attention_norm: nn.LayerNorm = nn.LayerNorm(width)
        self.position_wise: _GatedResidualNetwork = _GatedResidualNetwork(width, None, settings.dropout)
        self.output_gate: _GatedLinearUnit = _GatedLinearUnit(width)
        self.output_norm: nn.LayerNorm = nn.LayerNorm(width)
        self.head: nn.Linear = nn.Linear(width, 1)

    def forward(self, batch: CaseBatch) -> torch.Tensor:
        grid = grid_from_batch(batch)
        rows, steps = grid.observed.shape[0], grid.observed.shape[1]
        observed = grid.observed[:, :, : self.inputs]
        joined = torch.cat([grid.features[:, :, : self.inputs], observed], dim=-1)
        context = self.static_to_width(batch.static)
        selected, _ = self.static_variable(joined, context.unsqueeze(1).expand(rows, steps, self.width))

        live = (grid.times <= batch.index_hours.unsqueeze(-1)).unsqueeze(-1)
        encoded, _ = self.lstm(selected)
        encoded = encoded * live.to(encoded.dtype)
        enriched = self.enrichment(encoded, context.unsqueeze(1).expand(rows, steps, self.width))
        attended = self.attention(enriched, live.squeeze(-1))
        attended = self.attention_norm(enriched + self.attention_gate(attended))
        positioned: torch.Tensor = self.position_wise(attended)
        gated: torch.Tensor = self.output_norm(positioned + self.output_gate(positioned))
        position = grid.index_position
        rows_index = torch.arange(rows)
        final = gated[rows_index, position]
        logits: torch.Tensor = self.head(final).squeeze(-1)
        return logits

    def score(self, episodes: list[Episode]) -> ArmResult:
        self.eval()
        with torch.no_grad():
            batch = build_batch(episodes, capacity=self.capacity)
            probability = torch.sigmoid(self.forward(batch))
        return ArmResult(name=self.name, scores=probability)
