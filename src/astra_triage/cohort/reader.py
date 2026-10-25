"""Reading an episode table supplied by the caller.

Ref: Sec. 2.1 (the record), Table 1 (the cohort description).

The release ships no episode. The multi-site derivation cohort, the temporal
hold-out and the prospective arms are held by the participating institutions, and
the external resource is credentialed, so every command that needs records takes a
path from the caller. The file is one JSON object per line; the field list is in
``README.md`` under Data. A malformed row is reported with its line number and the
read stops, because a silently skipped episode changes the denominator of every
table.
"""

from __future__ import annotations

import json
from pathlib import Path

from astra_triage.cohort.errors import RecordError as RecordError
from astra_triage.cohort.split import SPLITS
from astra_triage.endpoints.adjudication import Decision, adjudicate, outcome_from
from astra_triage.endpoints.definition import Outcome, OutcomeKind, label
from astra_triage.streams.case import Episode, Observation
from astra_triage.streams.vitals import ClinicalItems, Vitals


def _as_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise RecordError(f"{name} is a number")
    return float(value)


def _as_items(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise RecordError(f"{name} is a list")
    return value


def _observations(payload: object, name: str) -> tuple[Observation, ...]:
    if payload is None:
        return ()
    if not isinstance(payload, list):
        raise RecordError(f"{name} must be a list of [hours, values] pairs")
    rows = []
    for entry in payload:
        if not isinstance(entry, list) or len(entry) != 2:
            raise RecordError(f"{name} entries are [hours, values]")
        hours, values = entry
        if not isinstance(values, list):
            raise RecordError(f"{name} values must be a list")
        rows.append(
            Observation(
                _as_float(hours, f"{name} hours"),
                tuple(_as_float(v, f"{name} value") for v in values),
            )
        )
    return tuple(rows)


def _vitals(payload: object) -> Vitals | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise RecordError("vitals must be an object")
    return Vitals(
        heart_rate=_as_float(payload["heart_rate"], "heart_rate"),
        respiratory_rate=_as_float(payload["respiratory_rate"], "respiratory_rate"),
        systolic_pressure=_as_float(payload["systolic_pressure"], "systolic_pressure"),
        temperature_c=_as_float(payload["temperature_c"], "temperature_c"),
        oxygen_saturation=_as_float(payload["oxygen_saturation"], "oxygen_saturation"),
        consciousness=str(payload.get("consciousness", "alert")),
    )


def _clinical(payload: object) -> ClinicalItems | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise RecordError("clinical items must be an object")
    return ClinicalItems(
        migration_to_right_lower_quadrant=bool(payload.get("migration", False)),
        anorexia=bool(payload.get("anorexia", False)),
        nausea_or_vomiting=bool(payload.get("nausea", False)),
        tenderness_right_lower_quadrant=bool(payload.get("tenderness", False)),
        rebound_or_cough_tenderness=bool(payload.get("rebound", False)),
        temperature_above_38=bool(payload.get("temperature_above_38", False)),
    )


def _split(payload: object) -> str:
    value = str(payload)
    if value not in SPLITS:
        raise RecordError(f"unknown split: {value}")
    return value


def _outcome(payload: dict[str, object]) -> int:
    """Read the adjudicated decision when the row carries the two blind reads."""
    reads = payload.get("reads")
    if reads is None:
        return _label(payload)
    if not isinstance(reads, list) or len(reads) != 2:
        raise RecordError("reads must hold the two blind decisions")
    arbiter = payload.get("arbiter")
    decision = adjudicate(
        Decision(str(reads[0])),
        Decision(str(reads[1])),
        Decision(str(arbiter)) if arbiter is not None else None,
    )
    hours = payload.get("reintervention_hours")
    outcome = outcome_from(
        decision,
        _as_float(hours, "reintervention_hours") if hours is not None else None,
        bool(payload.get("antibiotic_escalated", False)),
    )
    return label(outcome)


def _label(payload: dict[str, object]) -> int:
    if "label" in payload:
        raw = payload["label"]
        if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
            raise RecordError("a label is a number")
        return int(raw)
    kind = str(payload.get("outcome", "uneventful"))
    hours = payload.get("reintervention_hours")
    if kind == OutcomeKind.REINTERVENTION.value:
        if hours is None:
            raise RecordError("a re-intervention row needs its hours")
        recorded = _as_float(hours, "reintervention_hours")
        return label(Outcome(kind=OutcomeKind.REINTERVENTION, reintervention_hours=recorded))
    if kind == OutcomeKind.ANTIBIOTIC_ESCALATION.value:
        return label(Outcome(kind=OutcomeKind.ANTIBIOTIC_ESCALATION, antibiotic_escalated=True))
    return label(Outcome(kind=OutcomeKind.UNEVENTFUL))


def episode_from_row(row: dict[str, object]) -> Episode:
    try:
        return Episode(
            episode_id=str(row["episode_id"]),
            patient_id=str(row["patient_id"]),
            site=str(row["site"]),
            age_years=_as_float(row["age_years"], "age_years"),
            procedure=str(row["procedure"]),
            index_hours=_as_float(row["index_hours"], "index_hours"),
            ultrasound=_observations(row.get("ultrasound"), "ultrasound"),
            markers=_observations(row.get("markers"), "markers"),
            label=_outcome(row),
            split=_split(row.get("split", "train")),
            ultrasound_recorded=bool(row.get("ultrasound_recorded", True)),
            markers_recorded=bool(row.get("markers_recorded", True)),
            vitals=_vitals(row.get("vitals")),
            clinical=_clinical(row.get("clinical")),
            backgrounds=tuple(str(b) for b in _as_items(row.get("backgrounds", []), "backgrounds")),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RecordError(str(error)) from error


def read_episodes(path: str | Path) -> list[Episode]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(
            f"{source} does not exist; the episode table is private and must be staged by the caller"
        )
    episodes: list[Episode] = []
    with source.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as error:
                raise RecordError(f"line {number}: {error}") from error
            if not isinstance(payload, dict):
                raise RecordError(f"line {number}: a row is an object")
            try:
                episodes.append(episode_from_row(payload))
            except RecordError as error:
                raise RecordError(f"line {number}: {error}") from error
    if not episodes:
        raise RecordError(f"{source} holds no episode")
    return episodes


def write_episodes(episodes: list[Episode], path: str | Path) -> Path:
    """Write a table back, for staging checks and for the reader's own tooling."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for episode in episodes:
            payload = {
                "episode_id": episode.episode_id,
                "patient_id": episode.patient_id,
                "site": episode.site,
                "age_years": episode.age_years,
                "procedure": episode.procedure,
                "index_hours": episode.index_hours,
                "label": episode.label,
                "split": episode.split,
                "ultrasound_recorded": episode.ultrasound_recorded,
                "markers_recorded": episode.markers_recorded,
                "backgrounds": list(episode.backgrounds),
                "ultrasound": [[o.tau_hours, list(o.values)] for o in episode.ultrasound],
                "markers": [[o.tau_hours, list(o.values)] for o in episode.markers],
            }
            handle.write(json.dumps(payload) + "\n")
    return target
