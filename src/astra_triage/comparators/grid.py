"""Hourly resampling shared by the two recurrent comparators.

Ref: Table 2, rows "Temporal Fusion Transformer" and "GRU-D".

Both comparators expect a regular step. The two clinical streams arrive at
irregular, non-overlapping times, so the record is laid on an hourly grid anchored
at admission: the reading stamped inside an hour fills that hour, and every hour
without a reading is marked unobserved with the number of hours since the last
reading of the same stream. The grid stops at the latest index time in the batch,
so no comparator can see past its own decision point either.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from astra_triage.streams.case import CaseBatch, Episode, StreamBatch, build_batch
from astra_triage.streams.spec import marker_width, ultrasound_width

MAX_GRID_HOURS: int = 480


@dataclass(frozen=True)
class Grid:
    features: torch.Tensor
    observed: torch.Tensor
    delta: torch.Tensor
    times: torch.Tensor
    index_position: torch.Tensor

    @property
    def width(self) -> int:
        return int(self.features.shape[-1])


def hourly_grid(episodes: list[Episode], capacity: int = 32) -> Grid:
    batch = build_batch(episodes, capacity=capacity)
    return grid_from_batch(batch)


def grid_from_batch(batch: CaseBatch) -> Grid:
    rows = batch.index_hours.shape[0]
    length = int(torch.ceil(batch.index_hours.max()).item()) + 1
    length = max(1, min(length, MAX_GRID_HOURS))
    width = ultrasound_width() + marker_width()
    features = torch.zeros((rows, length, width), dtype=torch.float32)
    observed = torch.zeros((rows, length, width), dtype=torch.float32)
    delta = torch.zeros((rows, length, width), dtype=torch.float32)
    hours = torch.arange(length, dtype=torch.float32).unsqueeze(0).expand(rows, length)

    for row in range(rows):
        _place(batch.ultrasound, row, features, observed, 0, ultrasound_width(), length)
        _place(
            batch.markers,
            row,
            features,
            observed,
            ultrasound_width(),
            width,
            length,
        )
    for row in range(rows):
        _fill_delta(delta[row], observed[row], hours[row])
    positions = torch.clamp(torch.round(batch.index_hours).long(), max=length - 1)
    return Grid(
        features=features,
        observed=observed,
        delta=delta,
        times=hours,
        index_position=positions,
    )


def _place(
    stream: StreamBatch,
    row: int,
    features: torch.Tensor,
    observed: torch.Tensor,
    offset: int,
    width: int,
    length: int,
) -> None:
    values = stream.values[row]
    tau = stream.tau_hours[row]
    live = stream.mask[row]
    for slot in range(int(live.shape[0])):
        if not bool(live[slot]):
            continue
        hour = round(float(tau[slot]))
        if hour >= length:
            continue
        features[row, hour, offset : offset + width] = values[slot]
        observed[row, hour, offset : offset + width] = 1.0


def _fill_delta(delta: torch.Tensor, observed: torch.Tensor, hours: torch.Tensor) -> None:
    seen = torch.zeros(delta.shape[-1], dtype=torch.float32)
    last = torch.zeros(delta.shape[-1], dtype=torch.float32)
    for step in range(delta.shape[0]):
        now = hours[step]
        live = observed[step] > 0
        seen = torch.where(live, torch.ones_like(seen), seen)
        last = torch.where(live, now.expand_as(last), last)
        delta[step] = torch.where(seen > 0, now - last, now)
