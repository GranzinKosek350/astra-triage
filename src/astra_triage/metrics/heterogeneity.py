"""Between-site heterogeneity as I-squared.

Ref: Table 4, "Between-site heterogeneity, expressed as I 2 computed from
site-specific logit areas: 18.4% for ASTRA, 54.7% for the late-fusion baseline and
12.6% for last-value logistic regression."

The estimate is pooled on the logit scale, which is the scale on which an area
behaves like a mean. The between-site variance is the DerSimonian-Laird moment
estimator and I-squared is the share of total variation it accounts for. The
article does not name the variance estimator behind each site's standard error, so
the audit repeats the calculation under the closed-form Hanley-McNeil variance and
reports the number it obtains beside the number the table prints.
"""

from __future__ import annotations

import math

import numpy as np

from astra_triage.metrics.discrimination import hanley_mcneil_variance


def logit(area: float) -> float:
    clipped = min(max(area, 1e-6), 1.0 - 1e-6)
    return math.log(clipped / (1.0 - clipped))


def heterogeneity(estimates: list[float], standard_errors: list[float]) -> dict[str, float]:
    """Pooled logit area, Cochran's Q, the moment variance and I-squared."""
    if len(estimates) != len(standard_errors) or len(estimates) < 2:
        raise ValueError("heterogeneity needs at least two aligned site estimates")
    if any(se <= 0 for se in standard_errors):
        raise ValueError("a standard error must be positive")
    theta = np.array([logit(a) for a in estimates], dtype=np.float64)
    variance = np.square(np.array(standard_errors, dtype=np.float64))
    weights = 1.0 / variance
    pooled = float((weights * theta).sum() / weights.sum())
    q = float((weights * np.square(theta - pooled)).sum())
    degrees = len(theta) - 1
    scale = weights.sum() - float(np.square(weights).sum() / weights.sum())
    tau_squared = max(0.0, (q - degrees) / scale) if scale > 0 else 0.0
    i_squared = 0.0 if q <= 0 else max(0.0, (q - degrees) / q)
    return {
        "pooled_logit": pooled,
        "pooled_area": 1.0 / (1.0 + math.exp(-pooled)),
        "q": q,
        "degrees": float(degrees),
        "tau_squared": tau_squared,
        "i_squared": 100.0 * i_squared,
    }


def from_areas(areas: list[float], positives: list[int], negatives: list[int]) -> dict[str, float]:
    """The same calculation with the closed-form variance of each site's area.

    The closed form is on the area scale, so each standard error is carried to the
    logit scale by the delta method before pooling; pooling raw area-scale errors on
    logit estimates would overweight the sites with the largest areas.
    """
    errors = [
        math.sqrt(hanley_mcneil_variance(a, p, n)) / max(a * (1.0 - a), 1e-6)
        for a, p, n in zip(areas, positives, negatives, strict=True)
    ]
    return heterogeneity(areas, errors)
