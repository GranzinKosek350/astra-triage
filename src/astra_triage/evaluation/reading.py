"""The three-arm reading study.

Ref: Table 7 and Sec. 3.7, "Previously unexploited episodes of the series 240 were
evaluated by twenty-four health care professionals under three different
circumstances."

Each arm is a matrix of reader-by-episode probabilities: unaided, model alone (the
same score repeated for every reader), and supported. The summary reports the mean
of the per-reader areas, because pooling the readers would treat 24 reads of one
episode as 24 independent episodes. The two gaps the article prints are the
supported-minus-unaided and supported-minus-model differences, both computed from
the same episode resamples.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from astra_triage.metrics.bootstrap import RESAMPLES, paired_difference
from astra_triage.metrics.discrimination import auroc
from astra_triage.metrics.operating import at_threshold


@dataclass(frozen=True)
class ReadingArm:
    name: str
    probabilities: torch.Tensor
    median_minutes: float | None = None

    def __post_init__(self) -> None:
        if self.probabilities.ndim == 1:
            raise ValueError("a reading arm is a reader-by-episode matrix")


@dataclass(frozen=True)
class ReadingRow:
    condition: str
    auroc: float
    sensitivity: float
    specificity: float
    median_minutes: float | None


def per_reader_auroc(arm: ReadingArm, labels: torch.Tensor) -> float:
    values = []
    for reader in range(arm.probabilities.shape[0]):
        row = arm.probabilities[reader]
        if int(labels.sum()) == 0 or int(labels.sum()) == len(labels):
            continue
        values.append(auroc(row, labels))
    if not values:
        raise ValueError("no reader could be scored")
    return float(np.mean(values))


def summarise(arms: list[ReadingArm], labels: torch.Tensor) -> list[ReadingRow]:
    rows: list[ReadingRow] = []
    for arm in arms:
        pooled_sensitivity = []
        pooled_specificity = []
        for reader in range(arm.probabilities.shape[0]):
            point = at_threshold(arm.probabilities[reader], labels)
            pooled_sensitivity.append(point.sensitivity)
            pooled_specificity.append(point.specificity)
        rows.append(
            ReadingRow(
                condition=arm.name,
                auroc=per_reader_auroc(arm, labels),
                sensitivity=float(np.mean(pooled_sensitivity)),
                specificity=float(np.mean(pooled_specificity)),
                median_minutes=arm.median_minutes,
            )
        )
    return rows


def _mean_reader_metric(arm: ReadingArm, labels: torch.Tensor) -> Callable[[np.ndarray], float]:
    def inner(rows: np.ndarray) -> float:
        index = torch.as_tensor(rows)
        values = [
            auroc(arm.probabilities[reader][index], labels[index]) for reader in range(arm.probabilities.shape[0])
        ]
        return float(np.mean(values))

    return inner


def _gap(
    first: ReadingArm,
    second: ReadingArm,
    labels: torch.Tensor,
    resamples: int,
    seed: int,
) -> tuple[float, float, float, float]:
    return paired_difference(
        _mean_reader_metric(first, labels),
        _mean_reader_metric(second, labels),
        int(labels.numel()),
        resamples=resamples,
        seed=seed,
    )


def gaps(
    arms: list[ReadingArm],
    labels: torch.Tensor,
    resamples: int = RESAMPLES,
    seed: int = 20260825,
) -> dict[str, tuple[float, float, float, float]]:
    lookup = {arm.name: arm for arm in arms}
    if not {"clinician_alone", "model_alone", "clinician_with_support"} <= set(lookup):
        raise KeyError("the three reading conditions are named in the table")
    return {
        "supported_minus_unaided": _gap(
            lookup["clinician_with_support"], lookup["clinician_alone"], labels, resamples, seed
        ),
        "supported_minus_model": _gap(lookup["clinician_with_support"], lookup["model_alone"], labels, resamples, seed),
    }
