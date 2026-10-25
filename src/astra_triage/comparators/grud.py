"""GRU-D, the decaying-recurrent baseline of Table 2.

Ref: Table 2, row "GRU-D"; Sec. 3.3, "the decaying recurrent neural network of 3.5
by percentage points".

The model carries a hidden state across the hourly grid, decays both the last
observation and the hidden state by an amount that depends on how long each has gone
unrefreshed, and feeds the decayed input into the gate update. The decay rates are
learned per feature, so the network can decide that a two-day-old C-reactive protein
is still useful while a two-day-old wall thickness is not.

Each row stops at its own index hour: past that point the state is held, not
updated, so a case read early in the admission cannot borrow the next day's panel.
"""

from __future__ import annotations

import torch
from torch import nn

from astra_triage.comparators.base import ArmResult
from astra_triage.comparators.grid import grid_from_batch
from astra_triage.model.config import AstraConfig
from astra_triage.streams.case import CaseBatch, Episode, build_batch


class _DecayedCell(nn.Module):
    def __init__(self, width: int, hidden: int) -> None:
        super().__init__()
        self.input_decay: nn.Linear = nn.Linear(width, width)
        self.hidden_decay: nn.Linear = nn.Linear(1, hidden, bias=False)
        self.cell: nn.GRUCell = nn.GRUCell(width, hidden)

    def forward(
        self,
        values: torch.Tensor,
        observed: torch.Tensor,
        delta: torch.Tensor,
        state: torch.Tensor,
        last_value: torch.Tensor,
        mean_value: torch.Tensor,
    ) -> torch.Tensor:
        gamma_input = torch.exp(-torch.relu(self.input_decay(delta)))
        imputed = gamma_input * last_value + (1.0 - gamma_input) * mean_value
        values = observed * values + (1.0 - observed) * imputed
        gamma_hidden = torch.exp(-torch.relu(self.hidden_decay(delta.mean(dim=-1, keepdim=True))))
        updated: torch.Tensor = self.cell(values, gamma_hidden * state)
        return updated


class GRUD(nn.Module):
    name: str = "grud"
    learned: bool = True

    def __init__(self, config: AstraConfig | None = None, hidden: int = 64) -> None:
        super().__init__()
        settings = config if config is not None else AstraConfig()
        self.capacity: int = settings.capacity
        self.width: int = settings.ultrasound_width + settings.marker_width
        self.hidden: int = hidden
        self.cell: _DecayedCell = _DecayedCell(self.width, hidden)
        self.static: nn.Linear = nn.Linear(settings.static_width, hidden)
        self.head: nn.Sequential = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, settings.head_hidden),
            nn.GELU(),
            nn.Linear(settings.head_hidden, 1),
        )

    def forward(self, batch: CaseBatch) -> torch.Tensor:
        grid = grid_from_batch(batch)
        rows = grid.features.shape[0]
        state = torch.zeros((rows, self.hidden), dtype=grid.features.dtype)
        last_value = torch.zeros((rows, self.width), dtype=grid.features.dtype)
        mean_value = grid.features.mean(dim=(0, 1)).unsqueeze(0).expand(rows, self.width)
        for step in range(grid.features.shape[1]):
            values = grid.features[:, step]
            observed = grid.observed[:, step]
            last_value = observed * values + (1.0 - observed) * last_value
            proposed = self.cell(values, observed, grid.delta[:, step], state, last_value, mean_value)
            live = (step <= grid.index_position).unsqueeze(-1)
            state = torch.where(live, proposed, state)
        fused = state + self.static(batch.static)
        logits: torch.Tensor = self.head(fused).squeeze(-1)
        return logits

    def score(self, episodes: list[Episode]) -> ArmResult:
        self.eval()
        with torch.no_grad():
            batch = build_batch(episodes, capacity=self.capacity)
            probability = torch.sigmoid(self.forward(batch))
        return ArmResult(name=self.name, scores=probability)
