"""Episode records, index-time filtering and the padded batch layout.

Ref: Eq. (1), Sec. 2.3.0.1.

Timestamps are hours from admission to the surgical ward. The index time is the
hour at which the triage question is asked, and every observation later than the
index time is dropped before the model sees the case, so no path through this
module can read an observation the ward did not yet have.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import torch

from astra_triage.endpoints.definition import STALE_AFTER_HOURS, index_visible
from astra_triage.streams.spec import (
    MARKER_FIELDS,
    PROCEDURES,
    SITES,
    STATIC_CONTINUOUS,
    ULTRASOUND_FIELDS,
    Field,
    derive_marker_ratios,
    marker_width,
    static_width,
    transform,
    ultrasound_width,
)
from astra_triage.streams.vitals import ClinicalItems, Vitals

# A gap beyond a fortnight is indistinguishable from "no prior observation" once
# standardised, so it is capped rather than allowed to dominate the encoder.
GAP_CAP_HOURS: float = 14.0 * 24.0


@dataclass(frozen=True)
class Observation:
    tau_hours: float
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.tau_hours < 0:
            raise ValueError("observations are stamped after admission")


@dataclass(frozen=True)
class Episode:
    episode_id: str
    patient_id: str
    site: str
    age_years: float
    procedure: str
    index_hours: float
    ultrasound: tuple[Observation, ...]
    markers: tuple[Observation, ...]
    label: int
    split: str
    ultrasound_recorded: bool = True
    markers_recorded: bool = True
    vitals: Vitals | None = None
    clinical: ClinicalItems | None = None
    backgrounds: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.site not in SITES:
            raise ValueError(f"unknown site: {self.site}")
        if self.procedure not in PROCEDURES:
            raise ValueError(f"unknown procedure: {self.procedure}")
        if self.label not in (0, 1):
            raise ValueError("the endpoint is binary")
        for observation in self.ultrasound:
            if len(observation.values) != len(ULTRASOUND_FIELDS):
                raise ValueError("ultrasound descriptor width does not match the vocabulary")
        for observation in self.markers:
            if len(observation.values) != len(MARKER_FIELDS):
                raise ValueError("marker panel width does not match the vocabulary")

    def visible(self) -> Episode:
        """The case as it stood at the index time."""
        return replace(
            self,
            ultrasound=tuple(o for o in self.ultrasound if index_visible(o.tau_hours, self.index_hours)),
            markers=tuple(o for o in self.markers if index_visible(o.tau_hours, self.index_hours)),
        )


@dataclass(frozen=True)
class StreamBatch:
    values: torch.Tensor
    tau_hours: torch.Tensor
    gap_hours: torch.Tensor
    mask: torch.Tensor
    absent: torch.Tensor

    @property
    def width(self) -> int:
        return int(self.values.shape[-1])


@dataclass(frozen=True)
class CaseBatch:
    ultrasound: StreamBatch
    markers: StreamBatch
    static: torch.Tensor
    site: torch.Tensor
    label: torch.Tensor
    index_hours: torch.Tensor
    episode_ids: tuple[str, ...] = field(default_factory=tuple)

    def to(self, device: torch.device) -> CaseBatch:
        return CaseBatch(
            ultrasound=_move(self.ultrasound, device),
            markers=_move(self.markers, device),
            static=self.static.to(device),
            site=self.site.to(device),
            label=self.label.to(device),
            index_hours=self.index_hours.to(device),
            episode_ids=self.episode_ids,
        )


def _move(stream: StreamBatch, device: torch.device) -> StreamBatch:
    return StreamBatch(
        values=stream.values.to(device),
        tau_hours=stream.tau_hours.to(device),
        gap_hours=stream.gap_hours.to(device),
        mask=stream.mask.to(device),
        absent=stream.absent.to(device),
    )


def _gaps(times: list[float]) -> list[float]:
    """Hours since the previous distinct reading of the same stream.

    A second reading stamped at the same hour is not a new time point, so it carries
    the gap of the reading it repeats. This is what keeps the estimator count-blind:
    duplicating a reading duplicates an identical embedding, and the pooled vector is
    unchanged because the pooling weights split between the copies.
    """
    gaps: list[float] = []
    previous = 0.0
    previous_gap = 0.0
    for tau in times:
        if gaps and tau <= times[len(gaps) - 1]:
            gaps.append(previous_gap)
            continue
        previous_gap = min(tau - previous, GAP_CAP_HOURS)
        gaps.append(previous_gap)
        previous = tau
    return gaps


def _encode_stream(
    observations: tuple[Observation, ...],
    fields: tuple[Field, ...],
    width: int,
    capacity: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ordered = sorted(observations, key=lambda o: o.tau_hours)
    ordered = ordered[-capacity:] if len(ordered) > capacity else ordered
    values = torch.zeros((capacity, width), dtype=torch.float32)
    tau = torch.zeros(capacity, dtype=torch.float32)
    gap = torch.zeros(capacity, dtype=torch.float32)
    mask = torch.zeros(capacity, dtype=torch.bool)
    if ordered:
        raw = torch.tensor([o.values for o in ordered], dtype=torch.float32)
        encoded = transform(fields, raw)
        if encoded.shape[-1] < width:
            encoded = torch.cat([encoded, derive_marker_ratios(raw)], dim=-1)
        values[: len(ordered)] = encoded
        tau[: len(ordered)] = torch.tensor([o.tau_hours for o in ordered], dtype=torch.float32)
        gap[: len(ordered)] = torch.tensor(_gaps([o.tau_hours for o in ordered]), dtype=torch.float32)
        mask[: len(ordered)] = True
    return values, tau, gap, mask


def build_batch(episodes: list[Episode], capacity: int = 32) -> CaseBatch:
    """Pad the two streams of a set of episodes to a common length."""
    if not episodes:
        raise ValueError("an empty batch carries no information")
    batch = len(episodes)
    ultrasound_rows: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    marker_rows: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for episode in episodes:
        visible = episode.visible()
        if episode.ultrasound_recorded:
            ultrasound_rows.append(_encode_stream(visible.ultrasound, ULTRASOUND_FIELDS, ultrasound_width(), capacity))
        else:
            ultrasound_rows.append(
                (
                    torch.zeros((capacity, ultrasound_width())),
                    torch.zeros(capacity),
                    torch.zeros(capacity),
                    torch.zeros(capacity, dtype=torch.bool),
                )
            )
        if episode.markers_recorded:
            marker_rows.append(_encode_stream(visible.markers, MARKER_FIELDS, marker_width(), capacity))
        else:
            marker_rows.append(
                (
                    torch.zeros((capacity, marker_width())),
                    torch.zeros(capacity),
                    torch.zeros(capacity),
                    torch.zeros(capacity, dtype=torch.bool),
                )
            )

    static = torch.zeros((batch, static_width()), dtype=torch.float32)
    for row, episode in enumerate(episodes):
        age = torch.tensor([[episode.age_years]], dtype=torch.float32)
        static[row, 0] = transform(STATIC_CONTINUOUS, age).reshape(())
        static[row, 1 + PROCEDURES.index(episode.procedure)] = 1.0

    return CaseBatch(
        ultrasound=StreamBatch(
            values=torch.stack([row[0] for row in ultrasound_rows]),
            tau_hours=torch.stack([row[1] for row in ultrasound_rows]),
            gap_hours=torch.stack([row[2] for row in ultrasound_rows]),
            mask=torch.stack([row[3] for row in ultrasound_rows]),
            absent=torch.tensor([not e.ultrasound_recorded for e in episodes], dtype=torch.bool),
        ),
        markers=StreamBatch(
            values=torch.stack([row[0] for row in marker_rows]),
            tau_hours=torch.stack([row[1] for row in marker_rows]),
            gap_hours=torch.stack([row[2] for row in marker_rows]),
            mask=torch.stack([row[3] for row in marker_rows]),
            absent=torch.tensor([not e.markers_recorded for e in episodes], dtype=torch.bool),
        ),
        static=static,
        site=torch.tensor([SITES.index(e.site) for e in episodes], dtype=torch.long),
        label=torch.tensor([float(e.label) for e in episodes], dtype=torch.float32),
        index_hours=torch.tensor([e.index_hours for e in episodes], dtype=torch.float32),
        episode_ids=tuple(e.episode_id for e in episodes),
    )


def stale_flags(stream: StreamBatch, index_hours: torch.Tensor) -> torch.Tensor:
    """The 24-hour trigger of Sec. 2.3.0.3, as a per-case flag.

    A stream with no observation at all is not stale in this sense; it is absent,
    which is a different statement and carries a different embedding.
    """
    newest = torch.where(stream.mask, stream.tau_hours, torch.full_like(stream.tau_hours, -1e9))
    last = newest.max(dim=-1).values
    age = index_hours - last
    return stream.mask.any(dim=-1) & (age > STALE_AFTER_HOURS)
