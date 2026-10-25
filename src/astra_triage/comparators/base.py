"""The comparator interface shared by the seven rows of Table 2.

A comparator scores episodes. Learned arms build their own batch and expose their
parameters; rule arms read the record directly and carry a two-parameter logistic
link fitted on the derivation split so that their Brier, ECE and slope can be put
beside the model's. An arm that cannot read the record it needs returns a vector of
NaN and a reason, never a default score.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol

import torch

from astra_triage.streams.case import Episode


@dataclass
class ArmResult:
    name: str
    scores: torch.Tensor
    available: bool = True
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.available and self.reason is not None:
            raise ValueError("an available arm carries no reason")
        if not self.available and self.reason is None:
            raise ValueError("an unavailable arm states its reason")


class Scored(Protocol):
    """What every arm must offer: a name and a score over episodes."""

    name: str

    def score(self, episodes: list[Episode]) -> ArmResult:
        """Score every episode in the order it is given."""
        raise NotImplementedError


class Comparator:
    name: str = "comparator"
    learned: bool = False

    def fit(self, episodes: list[Episode]) -> None:
        """No-op for arms without free parameters."""
        return None

    def score(self, episodes: list[Episode]) -> ArmResult:
        raise NotImplementedError

    def state(self) -> dict[str, torch.Tensor]:
        return {}


class RuleUnavailable(LookupError):
    """The record does not carry the block the instrument reads."""


@dataclass
class Link:
    """Logistic link ``sigmoid(intercept + slope * raw)`` fitted on the derivation split."""

    intercept: float = 0.0
    slope: float = 1.0
    fitted: bool = False
    history: list[float] = field(default_factory=list)

    def __call__(self, raw: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.intercept + self.slope * raw)

    def fit(self, raw: torch.Tensor, target: torch.Tensor, iterations: int = 200) -> Link:
        if raw.numel() != target.numel():
            raise ValueError("the link is fitted on aligned scores and labels")
        design = torch.stack([torch.ones_like(raw), raw], dim=1).to(torch.float64)
        labels = target.to(torch.float64)
        beta = torch.zeros(2, dtype=torch.float64)
        beta[1] = 1.0
        ridge = 1e-6
        for _ in range(iterations):
            linear = design @ beta
            probability = torch.sigmoid(linear)
            weight = torch.clamp(probability * (1.0 - probability), min=1e-9)
            gradient = design.T @ (labels - probability) - ridge * beta
            hessian = (design * weight.unsqueeze(1)).T @ design
            hessian = hessian + ridge * torch.eye(2, dtype=torch.float64)
            step = torch.linalg.solve(hessian, gradient)
            beta = beta + step
            self.history.append(float(step.abs().max()))
            if float(step.abs().max()) < 1e-10:
                break
        self.intercept = float(beta[0])
        self.slope = float(beta[1])
        self.fitted = True
        return self


class LinkedRule(Comparator, ABC):
    """A clinical instrument mapped to a probability by a fitted two-parameter link.

    The instrument itself is deterministic, so the only thing fitted is the link. When
    the record cannot supply an item, the arm reports itself unavailable instead of
    substituting a normal value.
    """

    name: str = "rule"

    def __init__(self) -> None:
        self.link: Link = Link()

    @abstractmethod
    def totals(self, episodes: list[Episode]) -> torch.Tensor:
        raise NotImplementedError

    def fit(self, episodes: list[Episode]) -> None:
        """Fit the link when the record carries the block, and stay unfitted when it does not."""
        try:
            totals = self.totals(episodes)
        except RuleUnavailable:
            self.link = Link()
            return
        target = torch.tensor([float(e.label) for e in episodes], dtype=torch.float32)
        self.link.fit(totals, target)

    def score(self, episodes: list[Episode]) -> ArmResult:
        try:
            totals = self.totals(episodes)
        except RuleUnavailable as error:
            return ArmResult(
                name=self.name,
                scores=torch.full((len(episodes),), float("nan")),
                available=False,
                reason=str(error),
            )
        return ArmResult(name=self.name, scores=self.link(totals))
