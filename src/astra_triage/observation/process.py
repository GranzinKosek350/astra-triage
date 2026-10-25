"""The observation process that decides when a stream is measured.

Ref: Sec. 1, "sampling intensity is biased: the fact that a child is examined
often means that doctors are anxious about him or her", and the appendix
properties summarised in Sec. 2.3.0.1.

Sampling is a function of the latent state. When ``severity_slope`` is zero the
process is sequentially ignorable and measurement carries no information beyond
the values themselves; when it is positive, being measured is evidence and any
estimator that reads only the values inherits a bias.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SamplingProcess:
    base_logit: float = -0.30
    severity_slope: float = 0.0
    horizon_days: int = 10

    def __post_init__(self) -> None:
        if self.horizon_days < 1:
            raise ValueError("the observation horizon is at least one day")

    @property
    def ignorable(self) -> bool:
        return self.severity_slope == 0.0

    def daily_probability(self, severity: torch.Tensor) -> torch.Tensor:
        centred = severity - 0.5
        return torch.sigmoid(torch.full_like(severity, self.base_logit) + self.severity_slope * centred)

    def observed_days(self, severity: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
        """Bernoulli day mask from the daily intensity, ``[N, days]``."""
        probability = self.daily_probability(severity).unsqueeze(-1)
        draws = torch.rand((severity.shape[0], self.horizon_days), generator=rng, dtype=torch.float32)
        return draws < probability


@dataclass(frozen=True)
class PathDraw:
    """One draw from the model: a latent severity path and its two noisy read-outs."""

    severity: torch.Tensor
    marker: torch.Tensor
    ultrasound: torch.Tensor


def draw_paths(
    size: int,
    rng: torch.Generator,
    horizon_days: int = 10,
    rising_share: float = 0.5,
    noise: float = 0.35,
) -> PathDraw:
    """Latent severity paths with a marker and an ultrasound channel.

    Half the courses rise towards a late peak, which is the shape a developing
    collection makes; the other half decay from an early postoperative peak. The
    two channels differ in noisiness and in lag, which is what makes one of them
    more informative than the other when it is removed.
    """
    if not 0.0 <= rising_share <= 1.0:
        raise ValueError("the rising share is a probability")
    days = torch.arange(horizon_days, dtype=torch.float32)
    rising = torch.rand(size, generator=rng) < rising_share
    peak = 3.0 + 4.0 * torch.rand(size, generator=rng)
    base = torch.sigmoid((days.unsqueeze(0) - peak.unsqueeze(-1)) / 1.4)
    decay = torch.exp(-days.unsqueeze(0) / (2.0 + 4.0 * torch.rand(size, 1, generator=rng)))
    shape = torch.where(rising.unsqueeze(-1), base, decay)
    amplitude = 0.35 + 0.6 * torch.rand(size, 1, generator=rng)
    severity = torch.clamp(amplitude * shape, 0.0, 1.0)
    marker_noise = noise * torch.randn(severity.shape, generator=rng)
    lagged = torch.cat([severity[:, :1], severity[:, :-1]], dim=1)
    ultrasound_noise = 1.6 * noise * torch.randn(severity.shape, generator=rng)
    return PathDraw(
        severity=severity,
        marker=severity + marker_noise,
        ultrasound=0.6 * severity + 0.4 * lagged + ultrasound_noise,
    )
