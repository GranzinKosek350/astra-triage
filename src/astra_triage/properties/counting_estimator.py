"""Property 2: the counting estimator is calibrated inside the sample, and biased outside it.

Ref: Sec. 2.3.0.1, "the calibration consistency of the counting estimator under
sequential ignorability".

The counting estimator is the one the unconstrained arm can build: it predicts the
event rate observed among episodes with the same number of readings. Two statements
are separated here because they are easy to confuse.

Inside the sample the estimator stays close to the event rates of its count strata in
both regimes, and a held-out split of the same sample confirms it: the error sits near
two percentage points whether or not the process is ignorable. That is the consistency
the appendix names, and it holds because the sample is all the estimator is asked
about.

Transported to the population it is calibrated only when sampling is sequentially
ignorable. When the intensity of sampling depends on the state, the sample over-
represents the children who were measured often, and the transported risk is biased
by an amount that depends on the dependence and not on the number of episodes.

The draws come from the observation model; the population is the set of courses before
the observation filter.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from astra_triage.metrics.calibration import expected_calibration_error
from astra_triage.observation.process import SamplingProcess, draw_paths

COUNT_BINS: int = 5
MINIMUM_READINGS: int = 3


@dataclass(frozen=True)
class CountingResult:
    slope: float
    population_risk: float
    cohort_risk: float
    within_cohort_ece: float
    cohort_share: float
    episodes: int

    @property
    def bias(self) -> float:
        return self.cohort_risk - self.population_risk


def _stratified_predictions(
    counts: torch.Tensor, labels: torch.Tensor, fit: torch.Tensor, score: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    bins = torch.clamp((counts - MINIMUM_READINGS).clamp(min=0), max=COUNT_BINS - 1).long()
    base = labels[fit].mean()
    predictions = torch.full_like(labels, float(base))
    for value in range(COUNT_BINS):
        inside = fit & (bins == value)
        if int(inside.sum()) >= 2:
            predictions[bins == value] = labels[inside].mean()
    return predictions[score], labels[score]


def selection_bias(
    slope: float,
    episodes: int = 8000,
    seed: int = 3,
    horizon: int = 10,
    minimum_readings: int = MINIMUM_READINGS,
) -> CountingResult:
    rng = torch.Generator().manual_seed(seed)
    process = SamplingProcess(severity_slope=slope, horizon_days=horizon)
    paths = draw_paths(episodes, rng, horizon_days=horizon)
    intensity = process.daily_probability(paths.severity.reshape(-1)).reshape(paths.severity.shape)
    observed = torch.rand(paths.severity.shape, generator=rng) < intensity
    labels = (paths.severity[:, -1] > 0.5).to(torch.float32)
    counts = observed.sum(dim=1).to(torch.float32)
    cohort = counts >= minimum_readings
    if int(cohort.sum()) < 20:
        raise ValueError("the cohort filter left too few episodes")
    cohort_counts = counts[cohort]
    cohort_labels = labels[cohort]
    split = cohort_counts.numel() // 2
    fit = torch.zeros_like(cohort_labels, dtype=torch.bool)
    fit[:split] = True
    score = ~fit
    predicted, actual = _stratified_predictions(cohort_counts, cohort_labels, fit, score)
    return CountingResult(
        slope=slope,
        population_risk=float(labels.mean()),
        cohort_risk=float(cohort_labels.mean()),
        within_cohort_ece=expected_calibration_error(predicted, actual),
        cohort_share=float(cohort.to(torch.float32).mean()),
        episodes=int(cohort.sum()),
    )


def consistency_sweep(slopes: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 3.0), seed: int = 3) -> list[CountingResult]:
    return [selection_bias(slope, seed=seed) for slope in slopes]


def size_sweep(slope: float, sizes: tuple[int, ...] = (1000, 4000, 16000), seed: int = 3) -> list[CountingResult]:
    return [selection_bias(slope, episodes=size, seed=seed) for size in sizes]
