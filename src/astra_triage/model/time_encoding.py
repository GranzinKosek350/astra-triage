"""Continuous-time encoding of irregular observation stamps.

Ref: Sec. 2.3.0.1, "the time interval since the prior observation of each stream,
which has clinical significance", and Sec. 2.3.0.3.

Two clocks are encoded. The first is how long ago the reading was taken, measured
back from the index time, which is what makes a five-day-old ultrasound different
from this morning's C-reactive protein. The second is the position of the reading
inside the admission, which separates the early postoperative rise from a late
second wave. Both are Fourier features at dyadic periods from six hours to eight
days, plus the gap since the previous reading of the same stream.

Nothing here counts observations. A duplicated reading duplicates its own
timestamp and its own features, which is why pooling stays blind to the count.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from astra_triage.model.config import AstraConfig


class TimeEncoder(nn.Module):
    periods: torch.Tensor

    def __init__(self, config: AstraConfig) -> None:
        super().__init__()
        self.config: AstraConfig = config
        periods = torch.tensor(config.periods, dtype=torch.float32)
        self.register_buffer("periods", periods, persistent=True)
        self.project: nn.Linear | None
        if config.time_width:
            self.project = nn.Linear(config.time_width, config.d_model)
        else:
            self.project = None

    def features(self, tau_hours: torch.Tensor, gap_hours: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        age = torch.clamp(index.unsqueeze(-1) - tau_hours, min=0.0)
        angle_age = 2.0 * math.pi * age.unsqueeze(-1) / self.periods
        angle_position = 2.0 * math.pi * tau_hours.unsqueeze(-1) / self.periods
        gap_days = torch.clamp(gap_hours, min=0.0) / 24.0
        blocks = [
            torch.sin(angle_age),
            torch.cos(angle_age),
            torch.sin(angle_position),
            torch.cos(angle_position),
            gap_days.unsqueeze(-1),
            torch.log1p(gap_days).unsqueeze(-1),
        ]
        return torch.cat(blocks, dim=-1)

    def forward(self, tau_hours: torch.Tensor, gap_hours: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        if self.project is None:
            batch, length = tau_hours.shape
            return tau_hours.new_zeros((batch, length, 0))
        projected: torch.Tensor = self.project(self.features(tau_hours, gap_hours, index))
        return torch.tanh(projected)
