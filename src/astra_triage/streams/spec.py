"""Field vocabulary for the two observation streams and the static vector.

Ref: Sec. 1 (the two streams), Sec. 2.3.0.1 (Eq. 1), Table 1.

The method section of the article prints the stream names and the interval
argument but not a field list, so the fields are taken from the introduction,
which names bowel-wall thickness, peristalsis, fluid appearance, abscess
formation and Doppler behaviour for the ultrasound stream, and the white-cell
count, the neutrophil count, C-reactive protein, procalcitonin and the derived
ratios for the marker stream. The static vector is fixed by Eq. (1): age at
surgery, procedure type, site of surgery.

Each field carries a clinical plausibility band. Values outside the band are
clipped rather than rejected: a reading of 900 mg/L for C-reactive protein is a
recording convention, not a reason to drop the episode.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch


class Scale(StrEnum):
    CONTINUOUS = "continuous"
    LOG_CONTINUOUS = "log_continuous"
    ORDINAL = "ordinal"
    BINARY = "binary"


@dataclass(frozen=True)
class Field:
    name: str
    unit: str
    scale: Scale
    low: float
    high: float
    centre: float
    spread: float

    def __post_init__(self) -> None:
        if self.high <= self.low:
            raise ValueError(f"{self.name}: plausibility band is empty")
        if self.spread <= 0:
            raise ValueError(f"{self.name}: reference spread must be positive")


ULTRASOUND_FIELDS: tuple[Field, ...] = (
    Field("wall_thickness_mm", "mm", Scale.LOG_CONTINUOUS, 0.4, 12.0, 2.2, 0.7),
    Field("peristalsis_grade", "grade", Scale.ORDINAL, 0.0, 3.0, 1.5, 1.0),
    Field("fluid_echogenicity", "grade", Scale.ORDINAL, 0.0, 3.0, 0.9, 1.0),
    Field("fluid_volume_ml", "mL", Scale.LOG_CONTINUOUS, 0.0, 800.0, 8.0, 12.0),
    Field("abscess_present", "flag", Scale.BINARY, 0.0, 1.0, 0.15, 1.0),
    Field("doppler_grade", "grade", Scale.ORDINAL, 0.0, 2.0, 0.8, 1.0),
)

MARKER_FIELDS: tuple[Field, ...] = (
    Field("wbc_1e9_l", "1e9/L", Scale.LOG_CONTINUOUS, 0.2, 90.0, 11.5, 5.0),
    Field("neutrophil_1e9_l", "1e9/L", Scale.LOG_CONTINUOUS, 0.05, 80.0, 7.8, 4.2),
    Field("lymphocyte_1e9_l", "1e9/L", Scale.LOG_CONTINUOUS, 0.05, 30.0, 2.4, 1.6),
    Field("platelet_1e9_l", "1e9/L", Scale.LOG_CONTINUOUS, 10.0, 1200.0, 320.0, 120.0),
    Field("crp_mg_l", "mg/L", Scale.LOG_CONTINUOUS, 0.1, 500.0, 28.0, 40.0),
    Field("procalcitonin_ng_ml", "ng/mL", Scale.LOG_CONTINUOUS, 0.01, 200.0, 0.6, 1.8),
)

DERIVED_MARKER_NAMES: tuple[str, ...] = ("nlr", "plr", "nwr")

STATIC_CONTINUOUS: tuple[Field, ...] = (Field("age_at_surgery_years", "y", Scale.LOG_CONTINUOUS, 0.01, 18.0, 4.2, 4.3),)

PROCEDURES: tuple[str, ...] = (
    "resection_with_anastomosis",
    "enterostomy_formation_or_closure",
    "laparotomy_for_perforation",
    "other_bowel_procedure",
    "appendicectomy",
)

SITES: tuple[str, ...] = ("A", "B", "C", "D", "E")


def ultrasound_width() -> int:
    return len(ULTRASOUND_FIELDS)


def marker_width() -> int:
    return len(MARKER_FIELDS) + len(DERIVED_MARKER_NAMES)


def static_width() -> int:
    return len(STATIC_CONTINUOUS) + len(PROCEDURES)


def _centre_and_spread(field: Field, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.full_like(values, field.centre),
        torch.full_like(values, field.spread),
    )


def transform(fields: tuple[Field, ...], raw: torch.Tensor) -> torch.Tensor:
    """Clip to the plausibility band and put every field on its reference scale."""
    if raw.shape[-1] != len(fields):
        raise ValueError(f"expected {len(fields)} fields, received {raw.shape[-1]}")
    columns = []
    for index, field in enumerate(fields):
        column = raw[..., index]
        if field.scale in (Scale.CONTINUOUS, Scale.LOG_CONTINUOUS, Scale.ORDINAL):
            column = torch.clamp(column, field.low, field.high)
        else:
            column = torch.clamp(column, 0.0, 1.0)
        if field.scale is Scale.LOG_CONTINUOUS:
            column = torch.log1p(column)
        centre, spread = _centre_and_spread(field, column)
        columns.append((column - centre) / spread)
    return torch.stack(columns, dim=-1)


def inverse_transform(fields: tuple[Field, ...], standardised: torch.Tensor) -> torch.Tensor:
    """Reference-scale values back to the reported units, for the case read-out."""
    columns = []
    for index, field in enumerate(fields):
        column = standardised[..., index] * field.spread + field.centre
        if field.scale is Scale.LOG_CONTINUOUS:
            column = torch.expm1(column)
        columns.append(column)
    return torch.stack(columns, dim=-1)


def derive_marker_ratios(log_free: torch.Tensor) -> torch.Tensor:
    """Ratios computed by the attending team, reproduced from their inputs.

    Ref: Sec. 1, "as well as ratios derived from calculations done by medical
    professionals while attending the patient". The inputs arrive already on the
    log-continuous reference scale, so a ratio is a difference of exponentials;
    the ratio itself is formed in the reporting units and then put on the same
    reference scale.
    """
    if log_free.shape[-1] != len(MARKER_FIELDS):
        raise ValueError("ratio derivation needs the full marker panel")
    units = inverse_transform(MARKER_FIELDS, log_free)
    wbc = units[..., 0]
    neutrophil = units[..., 1]
    lymphocyte = units[..., 2]
    platelet = units[..., 3]
    floor = torch.full_like(lymphocyte, 0.05)
    nlr = neutrophil / torch.maximum(lymphocyte, floor)
    plr = platelet / torch.maximum(lymphocyte, floor)
    nwr = neutrophil / torch.maximum(wbc, floor)
    stack = torch.stack([nlr, plr, nwr], dim=-1)
    return torch.log1p(torch.clamp(stack, 0.0, 400.0)) / 2.0
