"""Atomic checkpoints that carry the seed.

A checkpoint is written to a temporary file beside its target and moved into place,
so an interrupted write cannot leave a half-serialised weight file behind. The seed
travels in the payload: a resumed run restores it before the first batch, otherwise
the data order and the augmentation draws would silently change.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from astra_triage.training.optim import Schedule, TrainConfig


@dataclass(frozen=True)
class ResumeState:
    epoch: int
    seed: int
    train_config: dict[str, Any]


def save(
    path: str | Path,
    module: nn.Module,
    config: TrainConfig,
    seed: int,
    epoch: int,
    optimizer: torch.optim.Optimizer | None = None,
    schedule: Schedule | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": module.state_dict(),
        "seed": seed,
        "epoch": epoch,
        "train_config": {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "precision": config.precision,
            "effective_batch": config.effective_batch,
        },
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if schedule is not None:
        payload["schedule"] = schedule.state()
    handle, temporary = tempfile.mkstemp(dir=target.parent, prefix=target.name, suffix=".tmp")
    os.close(handle)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target


def load(
    path: str | Path,
    module: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    schedule: Schedule | None = None,
) -> ResumeState:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"no checkpoint at {source}")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    module.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if schedule is not None and "schedule" in payload:
        schedule.load(payload["schedule"])
    return ResumeState(
        epoch=int(payload.get("epoch", 0)),
        seed=int(payload.get("seed", 0)),
        train_config=dict(payload.get("train_config", {})),
    )
