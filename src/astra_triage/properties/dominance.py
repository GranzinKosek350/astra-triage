"""Property 3: what an unconstrained estimator can earn is bounded by its nesting.

Ref: Sec. 2.3.0.1, "the limitation of how much can be earned by unconstrained
estimators"; Sec. 3.2, which states that a representation blind to the sampling levels
cannot beat one that is qualified to read them.

The two hypothesis classes are nested. The admissible class reads the mean of the
observed values and nothing else; the unconstrained class reads the same mean and, in
addition, the log count and the elapsed gap. Setting those two coefficients to zero
returns the admissible class, so the smallest achievable risk over the larger class
cannot exceed the smallest achievable risk over the smaller one. That is a statement
about optima and it holds for any cohort, which is why the check is run on draws from
the model and reported as a property rather than as a result.

The second number is the practical part: how much the nesting is worth. Under
sequential ignorability the count and the gap add little, and the gap between the two
fitted risks shrinks towards zero; under state-dependent sampling the larger class
earns more, and an admissible estimator cannot close that gap by being better tuned.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from astra_triage.metrics.discrimination import auroc
from astra_triage.observation.process import SamplingProcess, draw_paths

TOLERANCE: float = 1e-9


@dataclass(frozen=True)
class DominanceResult:
    slope: float
    admissible_risk: float
    unconstrained_risk: float
    admissible_area: float
    unconstrained_area: float
    nested: bool

    @property
    def earned(self) -> float:
        return self.admissible_risk - self.unconstrained_risk

    @property
    def area_gap(self) -> float:
        return self.unconstrained_area - self.admissible_area


def _fit(design: torch.Tensor, target: torch.Tensor, iterations: int = 400) -> torch.Tensor:
    beta = torch.zeros(design.shape[1])
    labels = target.to(design.dtype)
    for _ in range(iterations):
        linear = design @ beta
        mu = torch.sigmoid(linear)
        weight = torch.clamp(mu * (1.0 - mu), min=1e-9)
        gradient = design.T @ (labels - mu)
        hessian = (design * weight.unsqueeze(1)).T @ design
        step = torch.linalg.solve(hessian + 1e-8 * torch.eye(design.shape[1]), gradient)
        beta = beta + step
        if float(step.abs().max()) < 1e-11:
            break
    return beta


def _log_loss(logits: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.nn.functional.binary_cross_entropy_with_logits(logits, target.to(logits.dtype)))


def dominance(slope: float, episodes: int = 6000, seed: int = 9) -> DominanceResult:
    rng = torch.Generator().manual_seed(seed)
    process = SamplingProcess(severity_slope=slope, horizon_days=10)
    paths = draw_paths(episodes, rng, horizon_days=10)
    intensity = process.daily_probability(paths.severity.reshape(-1)).reshape(paths.severity.shape)
    observed = torch.rand(paths.severity.shape, generator=rng) < intensity
    observed[:, 0] = True
    carried = torch.zeros(episodes)
    last_day = torch.zeros(episodes, dtype=torch.long)
    running = torch.zeros(episodes)
    seen = torch.zeros(episodes)
    for day in range(observed.shape[1]):
        live = observed[:, day]
        carried = torch.where(live, paths.marker[:, day], carried)
        running = running + torch.where(live, paths.marker[:, day], torch.zeros_like(carried))
        seen = seen + live.to(torch.float32)
        last_day = torch.where(live, torch.full_like(last_day, day), last_day)
    mean_value = running / torch.clamp(seen, min=1.0)
    gap = (9 - last_day).to(torch.float32)
    log_count = torch.log1p(seen)
    target = (paths.severity[:, -1] > 0.5).to(torch.float32)
    ones = torch.ones(episodes)
    narrow = torch.stack([ones, mean_value], dim=1)
    wide = torch.stack([ones, mean_value, log_count, gap], dim=1)
    narrow_beta = _fit(narrow, target)
    wide_beta = _fit(wide, target)
    narrow_logits = narrow @ narrow_beta
    wide_logits = wide @ wide_beta
    embedded = torch.cat([narrow_beta, torch.zeros(2)])
    nested = abs(_log_loss(wide @ embedded, target) - _log_loss(narrow_logits, target)) <= 1e-6
    return DominanceResult(
        slope=slope,
        admissible_risk=_log_loss(narrow_logits, target),
        unconstrained_risk=_log_loss(wide_logits, target),
        admissible_area=auroc(narrow_logits, target),
        unconstrained_area=auroc(wide_logits, target),
        nested=bool(nested),
    )


def nesting_check(slope: float = 1.0, episodes: int = 2000, seed: int = 9) -> bool:
    result = dominance(slope, episodes, seed)
    return bool(result.nested and result.unconstrained_risk <= result.admissible_risk + TOLERANCE)
