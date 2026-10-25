"""Logistic regression on last-value features.

Ref: Table 2, row "Logistic regression, last-value features"; Sec. 1, "it considers
a report of an ultrasound that was obtained five days before and the C-reactive
protein level obtained this morning to be equally recent."

The arm is the one the introduction describes: each stream contributes its most
recent reading, the age of that reading is discarded, and a linear model is fitted
on the joined vector. It is the reference point that motivates the whole
representation, so it is implemented without any of the time machinery.
"""

from __future__ import annotations

import torch

from astra_triage.comparators.base import ArmResult
from astra_triage.model.ops import masked_last
from astra_triage.streams.case import Episode, Observation
from astra_triage.streams.spec import (
    MARKER_FIELDS,
    PROCEDURES,
    STATIC_CONTINUOUS,
    ULTRASOUND_FIELDS,
    Field,
    derive_marker_ratios,
    marker_width,
    static_width,
    transform,
    ultrasound_width,
)


def last_value_matrix(episodes: list[Episode]) -> torch.Tensor:
    """Last reading of each stream plus the static vector, in fixed order."""
    rows = []
    for episode in episodes:
        visible = episode.visible()
        ultrasound = _last(visible.ultrasound, ULTRASOUND_FIELDS, ultrasound_width())
        markers = _last(visible.markers, MARKER_FIELDS, marker_width())
        static = transform(STATIC_CONTINUOUS, torch.tensor([[episode.age_years]], dtype=torch.float32)).reshape(())
        one_hot = torch.zeros(len(PROCEDURES), dtype=torch.float32)
        one_hot[PROCEDURES.index(episode.procedure)] = 1.0
        rows.append(torch.cat([ultrasound, markers, static.view(1), one_hot]))
    return torch.stack(rows)


def _last(observations: tuple[Observation, ...], fields: tuple[Field, ...], width: int) -> torch.Tensor:
    if not observations:
        return torch.zeros(width, dtype=torch.float32)
    slots = len(observations)
    values = torch.tensor([o.values for o in observations], dtype=torch.float32)
    tau = torch.tensor([o.tau_hours for o in observations], dtype=torch.float32)
    encoded = transform(fields, values)
    if encoded.shape[-1] < width:
        encoded = torch.cat([encoded, derive_marker_ratios(values)], dim=-1)
    picked = masked_last(encoded.unsqueeze(0), torch.ones((1, slots), dtype=torch.bool), tau.unsqueeze(0))
    return picked.squeeze(0)


class LastValueLogistic(torch.nn.Module):
    name: str = "logistic_last_value"
    learned: bool = True
    closed_form: bool = True

    def __init__(self, l2: float = 1.0, iterations: int = 300) -> None:
        super().__init__()
        width = ultrasound_width() + marker_width() + static_width()
        self.width: int = width
        self.l2: float = l2
        self.iterations: int = iterations
        self.linear: torch.nn.Linear = torch.nn.Linear(width, 1)
        self.registered: bool = False

    def fit_matrix(self, features: torch.Tensor, target: torch.Tensor) -> None:
        design = torch.cat([torch.ones((features.shape[0], 1)), features], dim=1).to(torch.float64)
        labels = target.to(torch.float64)
        beta = torch.zeros(design.shape[1], dtype=torch.float64)
        penalty = torch.eye(design.shape[1], dtype=torch.float64) * self.l2
        penalty[0, 0] = 0.0
        for _ in range(self.iterations):
            linear = design @ beta
            probability = torch.sigmoid(linear)
            weight = torch.clamp(probability * (1.0 - probability), min=1e-8)
            gradient = design.T @ (labels - probability) - penalty @ beta
            hessian = (design * weight.unsqueeze(1)).T @ design + penalty
            step = torch.linalg.solve(hessian, gradient)
            beta = beta + step
            if float(step.abs().max()) < 1e-10:
                break
        with torch.no_grad():
            self.linear.weight.copy_(beta[1:].to(torch.float32).view(1, -1))
            self.linear.bias.copy_(beta[0].to(torch.float32).view(()))
        self.registered = True

    def fit(self, episodes: list[Episode]) -> None:
        features = last_value_matrix(episodes)
        target = torch.tensor([float(e.label) for e in episodes], dtype=torch.float32)
        self.fit_matrix(features, target)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.linear(features).squeeze(-1)
        return logits

    def score(self, episodes: list[Episode]) -> ArmResult:
        self.eval()
        with torch.no_grad():
            probability = torch.sigmoid(self.forward(last_value_matrix(episodes)))
        return ArmResult(name=self.name, scores=probability)
