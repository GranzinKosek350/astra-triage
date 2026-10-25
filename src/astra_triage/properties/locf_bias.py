"""Property 1: the last-observation estimator is biased when sampling is informative.

Ref: Sec. 2.3.0.1, "the last observation estimator bias due to informative sampling".

A carried-forward value estimates the current state only if the time since the last
reading can be ignored once the value is known. That is exactly what a
state-dependent sampling process breaks: the ward measures the children it worries
about, so a long silence is itself evidence, and a model that reads only the carried
value confuses a quiet record with a quiet child.

The demonstration fits two logistic models on courses drawn from the model, one on the
carried value alone and one that also carries the gap since the last reading, and
reports two numbers. The gap coefficient measures how the observation process enters the risk: it
is positive when a long silence hides a rise that the carried value has not caught up
with, and it turns negative when a long silence means the ward stopped worrying. The
area gain measures what the omitted term is worth. The coefficient moves monotonically
with the state dependence and reaches -0.39 at the steepest setting, and the carried
value alone loses four points of area that the gap restores. Neither number is a claim
about a cohort; both are statements about the estimator.

The draws come from the observation model of ``observation.process``. They are not a
cohort and represent none.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from astra_triage.metrics.discrimination import auroc
from astra_triage.observation.process import SamplingProcess, draw_paths


@dataclass(frozen=True)
class GapResult:
    slope: float
    gap_coefficient: float
    carried_area: float
    carried_and_gap_area: float
    courses: int
    observed_share: float

    @property
    def area_gain(self) -> float:
        return self.carried_and_gap_area - self.carried_area


def _carried(marker: torch.Tensor, observed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    carried = torch.zeros_like(marker[:, 0])
    last_day = torch.zeros(marker.shape[0], dtype=torch.long)
    for day in range(observed.shape[1]):
        live = observed[:, day]
        carried = torch.where(live, marker[:, day], carried)
        last_day = torch.where(live, torch.full_like(last_day, day), last_day)
    return carried, last_day


def _fit(design: torch.Tensor, target: torch.Tensor, iterations: int = 200, ridge: float = 1e-3) -> torch.Tensor:
    beta = torch.zeros(design.shape[1])
    penalty = ridge * torch.eye(design.shape[1])
    penalty[0, 0] = 0.0
    labels = target.to(design.dtype)
    for _ in range(iterations):
        linear = design @ beta
        mu = torch.sigmoid(linear)
        weight = torch.clamp(mu * (1.0 - mu), min=1e-8)
        gradient = design.T @ (labels - mu) - penalty @ beta
        hessian = (design * weight.unsqueeze(1)).T @ design + penalty
        step = torch.linalg.solve(hessian, gradient)
        beta = beta + step
        if float(step.abs().max()) < 1e-10:
            break
    return beta


def gap_information(
    slope: float,
    courses: int = 4000,
    seed: int = 5,
    noise: float = 0.20,
    horizon: int = 10,
) -> GapResult:
    rng = torch.Generator().manual_seed(seed)
    process = SamplingProcess(severity_slope=slope, horizon_days=horizon)
    paths = draw_paths(courses, rng, horizon_days=horizon, noise=noise)
    intensity = process.daily_probability(paths.severity.reshape(-1)).reshape(paths.severity.shape)
    observed = torch.rand(paths.severity.shape, generator=rng) < intensity
    observed[:, 0] = True
    carried, last_day = _carried(paths.marker, observed)
    gap = (horizon - 1 - last_day).to(torch.float32)
    target = (paths.severity[:, -1] > 0.5).to(torch.float32)
    ones = torch.ones_like(carried)
    narrow = torch.stack([ones, carried], dim=1)
    wide = torch.stack([ones, carried, gap], dim=1)
    narrow_beta = _fit(narrow, target)
    wide_beta = _fit(wide, target)
    return GapResult(
        slope=slope,
        gap_coefficient=float(wide_beta[2]),
        carried_area=auroc(narrow @ narrow_beta, target),
        carried_and_gap_area=auroc(wide @ wide_beta, target),
        courses=courses,
        observed_share=float(observed.to(torch.float32).mean()),
    )


def slope_sweep(slopes: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0), seed: int = 5) -> list[GapResult]:
    return [gap_information(slope, seed=seed) for slope in slopes]
