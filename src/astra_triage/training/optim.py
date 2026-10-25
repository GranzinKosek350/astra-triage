"""Optimiser, schedule and the run-level settings.

The article prints no training schedule: Sec. 2.4, "Reporting", is a heading with no
body, and no table of hyperparameters appears anywhere in the text or the
supplementary tables. Every value in ``TrainConfig`` is therefore an engineering
default, recorded as such in the verification report, and the README states the
arithmetic that follows from them rather than an expected score.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 40
    batch_size: int = 32
    grad_accumulation: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 500
    scheduler: str = "cosine"
    precision: str = "bf16"
    grad_clip: float = 1.0
    ema_decay: float = 0.0
    seeds: tuple[int, ...] = (20260825, 20260826, 20260827, 20260828, 20260829)
    world_size: int = 1
    lambda_cal: float = 0.50
    lambda_site: float = 0.10
    reversal_ramp_epochs: int = 5

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.grad_accumulation < 1:
            raise ValueError("epochs and batch sizes are positive")
        if self.precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError(f"unknown precision: {self.precision}")
        if self.scheduler not in {"cosine", "constant"}:
            raise ValueError(f"unknown schedule: {self.scheduler}")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("the EMA decay sits in [0, 1)")

    @property
    def effective_batch(self) -> int:
        return self.batch_size * self.grad_accumulation * self.world_size


def build_optimizer(module: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    parameters = [p for p in module.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("the arm carries no trainable parameter")
    return torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)


def schedule_scale(step: int, total: int, warmup: int) -> float:
    """Linear warm-up into a cosine decay, as a multiplier on the base rate."""
    if total <= 0:
        return 1.0
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


class Schedule:
    def __init__(self, optimizer: torch.optim.Optimizer, config: TrainConfig, total: int) -> None:
        self.optimizer: torch.optim.Optimizer = optimizer
        self.config: TrainConfig = config
        self.total: int = total
        self.base: list[float] = [group["lr"] for group in optimizer.param_groups]
        self.step: int = 0

    def advance(self) -> float:
        scale = (
            1.0
            if self.config.scheduler == "constant"
            else schedule_scale(self.step, self.total, self.config.warmup_steps)
        )
        for group, base in zip(self.optimizer.param_groups, self.base, strict=True):
            group["lr"] = base * scale
        self.step += 1
        return scale

    def state(self) -> dict[str, object]:
        return {"step": self.step, "base": self.base}

    def load(self, state: dict[str, object]) -> None:
        step = state["step"]
        if not isinstance(step, int):
            raise ValueError("a restored schedule needs an integer step")
        self.step = step
        base = state["base"]
        if isinstance(base, list):
            self.base = [float(value) for value in base]


class ModelEMA:
    """A shadow copy kept at a decayed average, skipped when the decay is zero."""

    def __init__(self, module: nn.Module, decay: float) -> None:
        self.decay: float = decay
        self.enabled: bool = decay > 0.0
        self.shadow: dict[str, torch.Tensor] = {
            name: tensor.detach().clone() for name, tensor in module.state_dict().items()
        }

    def update(self, module: nn.Module) -> None:
        if not self.enabled:
            return
        for name, tensor in module.state_dict().items():
            if not tensor.is_floating_point():
                self.shadow[name] = tensor.detach().clone()
                continue
            self.shadow[name].mul_(self.decay).add_(tensor.detach(), alpha=1.0 - self.decay)

    def copy_to(self, module: nn.Module) -> None:
        module.load_state_dict(self.shadow)


def average_gradients(parameters: Iterable[torch.nn.Parameter], scale: float) -> None:
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.mul_(scale)
