"""Calibration: expected calibration error, Brier score, slope and intercept.

Ref: Table 2 columns ECE, Brier and Slope; Sec. 3.2, "the calibration slope found
from 0.96 (95% CI 0.90 to 1.02) is used together with the intercept value from
-0.04 (-0.21 to 0.13)".

The slope is the coefficient of a logistic recalibration of the outcome on the
logit of the prediction. A slope of one with an intercept of zero is the statement
that the probabilities are already on the right scale; a slope below one is the
over-confidence the external table reports.
"""

from __future__ import annotations

import math

import numpy as np
import torch

ECE_BINS: int = 10


def _arrays(probability: torch.Tensor, target: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    p = probability.detach().to(torch.float64).cpu().numpy().reshape(-1)
    y = target.detach().to(torch.float64).cpu().numpy().reshape(-1)
    if p.shape != y.shape:
        raise ValueError("probabilities and labels must align")
    if np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("probabilities sit in [0, 1]")
    return p, y


def brier(probability: torch.Tensor, target: torch.Tensor) -> float:
    p, y = _arrays(probability, target)
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(probability: torch.Tensor, target: torch.Tensor, bins: int = ECE_BINS) -> float:
    """Equal-width binning, which is the convention the article's ECE follows."""
    p, y = _arrays(probability, target)
    edges = np.linspace(0.0, 1.0, bins + 1)
    membership = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for index in range(bins):
        live = membership == index
        if not live.any():
            continue
        total += live.mean() * abs(p[live].mean() - y[live].mean())
    return float(total)


def reliability_table(
    probability: torch.Tensor, target: torch.Tensor, bins: int = ECE_BINS
) -> list[tuple[float, float, float, int]]:
    p, y = _arrays(probability, target)
    edges = np.linspace(0.0, 1.0, bins + 1)
    membership = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows: list[tuple[float, float, float, int]] = []
    for index in range(bins):
        live = membership == index
        count = int(live.sum())
        if count == 0:
            continue
        rows.append((float(p[live].mean()), float(y[live].mean()), count / p.size, count))
    return rows


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """A stable logistic, so a degenerate recalibration cannot overflow."""
    positive = x >= 0
    result = np.empty_like(x)
    result[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exponent = np.exp(x[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    return result


def _logit(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.asarray(np.log(clipped / (1.0 - clipped)), dtype=np.float64)


def calibration_slope(probability: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    """Intercept and slope of the logistic recalibration, by Newton iterations."""
    p, y = _arrays(probability, target)
    x = _logit(p)
    design = np.stack([np.ones_like(x), x], axis=1)
    if float(np.std(x)) < 1e-6:
        return float("nan"), float("nan")
    beta = np.array([0.0, 1.0], dtype=np.float64)
    ridge = np.array([[1e-8, 0.0], [0.0, 1e-8]], dtype=np.float64)
    for _ in range(100):
        linear = design @ beta
        mu = _sigmoid(linear)
        weight = np.clip(mu * (1.0 - mu), 1e-9, None)
        gradient = design.T @ (y - mu) - ridge @ beta
        hessian = (design * weight[:, None]).T @ design + ridge
        step = np.linalg.solve(hessian, gradient)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    if not np.all(np.isfinite(beta)) or abs(float(beta[1])) > 100.0:
        return float("nan"), float("nan")
    return float(beta[0]), float(beta[1])


def slope_standard_error(probability: torch.Tensor, target: torch.Tensor) -> float:
    """Wald standard error of the recalibration slope."""
    p, _ = _arrays(probability, target)
    x = _logit(p)
    design = np.stack([np.ones_like(x), x], axis=1)
    intercept, slope = calibration_slope(probability, target)
    beta = np.array([intercept, slope], dtype=np.float64)
    linear = design @ beta
    mu = _sigmoid(linear)
    weight = np.clip(mu * (1.0 - mu), 1e-9, None)
    hessian = (design * weight[:, None]).T @ design
    covariance = np.linalg.pinv(hessian)
    return float(math.sqrt(max(covariance[1, 1], 0.0)))


def murphy_components(probability: torch.Tensor, target: torch.Tensor, bins: int = ECE_BINS) -> dict[str, float]:
    """Reliability, resolution and uncertainty, with the Brier identity as the check."""
    _, y = _arrays(probability, target)
    base = y.mean()
    uncertainty = float(base * (1.0 - base))
    reliability = 0.0
    resolution = 0.0
    for mean_p, mean_y, share, _ in reliability_table(probability, target, bins):
        reliability += share * (mean_p - mean_y) ** 2
        resolution += share * (mean_y - base) ** 2
    return {
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": uncertainty,
        "brier": float(reliability - resolution + uncertainty),
    }
