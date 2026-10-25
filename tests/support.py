"""Episode objects built inside the test process.

They are tensor inputs for the checks, not a cohort: they carry no clinical claim and no
reported number is derived from them.
"""

from __future__ import annotations

from astra_triage.streams.case import Episode, Observation
from astra_triage.streams.vitals import ClinicalItems, Vitals


def make_episode(
    index: int,
    label: int,
    site: str = "A",
    ultrasound: tuple[float, ...] = (8.0, 32.0),
    markers: tuple[float, ...] = (4.0, 28.0, 52.0),
    age: float = 3.0,
    procedure: str = "resection_with_anastomosis",
    index_hours: float = 60.0,
    backgrounds: tuple[str, ...] = (),
    vitals: bool = False,
    clinical: bool = False,
    ultrasound_recorded: bool = True,
    markers_recorded: bool = True,
) -> Episode:
    ultrasound_rows = tuple(
        Observation(t, (1.8 + 0.4 * index, float(index % 3), 1.0, 4.0 + index, 0.0, 1.0)) for t in ultrasound
    )
    marker_rows = tuple(
        Observation(
            t,
            (10.0 + index, 7.0 + 0.5 * index, 2.2, 280.0, 30.0 + 5.0 * t, 0.6 + 0.1 * t),
        )
        for t in markers
    )
    return Episode(
        episode_id=f"e{index}",
        patient_id=f"p{index // 2}",
        site=site,
        age_years=age,
        procedure=procedure,
        index_hours=index_hours,
        ultrasound=ultrasound_rows,
        markers=marker_rows,
        label=label,
        split="train",
        vitals=Vitals(120.0, 34.0, 95.0, 37.2, 97.0) if vitals else None,
        clinical=ClinicalItems(True, True, False, True, False, False) if clinical else None,
        backgrounds=backgrounds,
        ultrasound_recorded=ultrasound_recorded,
        markers_recorded=markers_recorded,
    )
