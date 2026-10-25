"""The training loop.

Batching is over episodes, one epoch is one pass in a seeded order, and the
adversarial strength follows its ramp across the first epochs. The fused arm
optimises Eq. (3); a comparator optimises the class-weighted cross-entropy alone,
because the two penalty terms exist to shape the fused representation and have no
counterpart in the baseline designs.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from astra_triage.model.astra import Astra
from astra_triage.objectives.composite import LossWeights, binary_cross_entropy, composite_loss
from astra_triage.objectives.site_loss import ramp
from astra_triage.streams.case import Episode, build_batch
from astra_triage.training.optim import ModelEMA, Schedule, TrainConfig
from astra_triage.training.seeding import rng


@dataclass
class History:
    epochs: list[dict[str, float]] = field(default_factory=list)
    batches: int = 0

    def append(self, row: dict[str, float]) -> None:
        self.epochs.append(row)

    @property
    def first_loss(self) -> float:
        return self.epochs[0]["total"] if self.epochs else float("nan")

    @property
    def last_loss(self) -> float:
        return self.epochs[-1]["total"] if self.epochs else float("nan")


def _class_counts(rows: list[Episode]) -> tuple[int, int]:
    """Class counts over the whole training set, not over one batch.

    The effective-number weights describe the cohort, so they are computed once; a
    batch that happens to hold one class then still gets the cohort weighting instead
    of an undefined ratio.
    """
    positives = sum(1 for row in rows if row.label)
    return len(rows) - positives, positives


def _context(device: torch.device, precision: str) -> Any:
    if precision == "fp32" or device.type not in {"cuda", "cpu"}:
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def train_arm(
    arm: nn.Module,
    episodes: list[Episode],
    config: TrainConfig,
    device: torch.device,
    seed: int | None = None,
    capacity: int = 32,
    steps_limit: int | None = None,
) -> History:
    rows = list(episodes)
    run_seed = int(config.seeds[0] if seed is None else seed)
    order = rng(run_seed)
    trainable = [p for p in arm.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=config.weight_decay)
    batches_per_epoch = max(1, (len(rows) + config.batch_size - 1) // config.batch_size)
    schedule = Schedule(optimizer, config, config.epochs * batches_per_epoch)
    ema = ModelEMA(arm, config.ema_decay)
    weights = LossWeights(lambda_cal=config.lambda_cal, lambda_site=config.lambda_site)
    negatives, positives = _class_counts(rows)
    if positives == 0 or negatives == 0:
        raise ValueError("the training set carries a single class")
    history = History()
    arm.train()
    stopped = False
    for epoch in range(config.epochs):
        if stopped:
            break
        permutation = torch.randperm(len(rows), generator=order).tolist()
        totals = {"total": 0.0, "bce": 0.0, "calibration": 0.0, "site": 0.0}
        seen = 0
        strength = 1.0
        if isinstance(arm, Astra):
            strength = ramp(epoch, max(config.reversal_ramp_epochs, 1)) if epoch < config.reversal_ramp_epochs else 1.0
            arm.set_reversal_strength(strength)
        optimizer.zero_grad(set_to_none=True)
        for batch_index in range(batches_per_epoch):
            index = permutation[batch_index * config.batch_size : (batch_index + 1) * config.batch_size]
            if not index:
                continue
            batch = build_batch([rows[i] for i in index], capacity=capacity).to(device)
            with _context(device, config.precision):
                output = arm(batch)
                if isinstance(arm, Astra):
                    breakdown = composite_loss(
                        output.logits,
                        output.probability,
                        batch.label,
                        output.site_logits,
                        batch.site,
                        (negatives, positives),
                        weights,
                    )
                    loss = breakdown.total
                    parts = breakdown.as_floats()
                else:
                    logits: torch.Tensor = output
                    loss = binary_cross_entropy(logits, batch.label, negatives, positives)
                    parts = {
                        "total": float(loss.detach()),
                        "bce": float(loss.detach()),
                        "calibration": 0.0,
                        "site": 0.0,
                    }
            torch.autograd.backward(loss / config.grad_accumulation)
            if (batch_index + 1) % config.grad_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(trainable, config.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                ema.update(arm)
            schedule.advance()
            history.batches += 1
            for key in totals:
                totals[key] += parts[key] * len(index)
            seen += len(index)
            if steps_limit is not None and history.batches >= steps_limit:
                stopped = True
                break
        for key in totals:
            totals[key] = totals[key] / max(seen, 1)
        totals["learning_rate"] = float(optimizer.param_groups[0]["lr"])
        totals["reversal"] = strength
        history.append(totals)
    if ema.enabled:
        ema.copy_to(arm)
    return history


def predict(arm: nn.Module, episodes: list[Episode], capacity: int = 32) -> torch.Tensor:
    """Scores in evaluation mode, without gradients."""
    arm.eval()
    with torch.no_grad():
        batch = build_batch(episodes, capacity=capacity)
        output = arm(batch)
        logits: torch.Tensor = output.logits if isinstance(arm, Astra) else output
        probability = torch.sigmoid(logits)
    return probability
