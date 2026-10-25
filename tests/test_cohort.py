"""Episode tables, splits and subgroups."""

from __future__ import annotations

import pytest

from astra_triage.cohort.reader import (
    RecordError,
    episode_from_row,
    read_episodes,
    write_episodes,
)
from astra_triage.cohort.split import (
    assert_patient_disjoint,
    event_count,
    patient_overlap,
    select,
    site_distribution,
)
from astra_triage.cohort.strata import STRATA, describe, members, sparsity_of
from astra_triage.streams.case import Episode
from support import make_episode


def _row(episode: Episode) -> dict[str, object]:
    return {
        "episode_id": episode.episode_id,
        "patient_id": episode.patient_id,
        "site": episode.site,
        "age_years": episode.age_years,
        "procedure": episode.procedure,
        "index_hours": episode.index_hours,
        "label": episode.label,
        "split": episode.split,
        "ultrasound": [[o.tau_hours, list(o.values)] for o in episode.ultrasound],
        "markers": [[o.tau_hours, list(o.values)] for o in episode.markers],
    }


def test_round_trip_through_a_file(tmp_path) -> None:
    rows = [make_episode(i, i % 2) for i in range(4)]
    path = tmp_path / "cohort.jsonl"
    write_episodes(rows, path)
    restored = read_episodes(path)
    assert [e.episode_id for e in restored] == [e.episode_id for e in rows]
    assert restored[0].ultrasound[0].tau_hours == rows[0].ultrasound[0].tau_hours


def test_missing_file_names_the_reason(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        read_episodes(tmp_path / "absent.jsonl")


def test_malformed_row_reports_its_line(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"episode_id": "x"}\n', encoding="utf-8")
    with pytest.raises(RecordError) as error:
        read_episodes(path)
    assert "line 1" in str(error.value)


def test_unknown_split_is_rejected() -> None:
    row = _row(make_episode(1, 0))
    row["split"] = "holdout"
    with pytest.raises(RecordError):
        episode_from_row(row)


def test_adjudicated_reads_define_the_label() -> None:
    row = _row(make_episode(1, 0))
    row.pop("label")
    row["reads"] = ["reintervention", "reintervention"]
    row["reintervention_hours"] = 30.0
    assert episode_from_row(row).label == 1


def test_discordant_reads_need_an_arbiter() -> None:
    row = _row(make_episode(1, 0))
    row.pop("label")
    row["reads"] = ["reintervention", "no_reintervention"]
    with pytest.raises(RecordError):
        episode_from_row(row)


def test_patient_overlap_is_detected() -> None:
    first = make_episode(1, 0)
    second = make_episode(1, 1)
    second = Episode(
        episode_id="e2",
        patient_id=first.patient_id,
        site="B",
        age_years=2.0,
        procedure="appendicectomy",
        index_hours=40.0,
        ultrasound=second.ultrasound,
        markers=second.markers,
        label=1,
        split="test",
    )
    overlap = patient_overlap([first, second])
    assert first.patient_id in overlap
    with pytest.raises(RecordError):
        assert_patient_disjoint([first, second])


def test_select_returns_the_partition() -> None:
    episodes = [make_episode(i, i % 2) for i in range(4)]
    assert len(select(episodes, "train")) == 4
    with pytest.raises(RecordError):
        select(episodes, "external")


def test_event_count_and_site_distribution() -> None:
    episodes = [make_episode(i, i % 2, site="A" if i < 2 else "B") for i in range(4)]
    events, share = event_count(episodes)
    assert events == 2
    assert share == pytest.approx(50.0)
    assert site_distribution(episodes) == {"A": 2, "B": 2}


def test_subgroups_partition_by_age() -> None:
    episodes = [make_episode(i, 0, age=age) for i, age in enumerate((0.5, 3.0, 8.0, 15.0))]
    assert len(members(episodes, "age_under_1")) == 1
    assert len(members(episodes, "age_1_to_5")) == 1
    assert len(members(episodes, "age_6_to_12")) == 1
    assert len(members(episodes, "age_13_to_17")) == 1


def test_background_subgroups_need_the_record() -> None:
    with_background = make_episode(1, 0, backgrounds=("enterostomy",))
    without = make_episode(2, 0)
    assert members([with_background, without], "enterostomy") == [with_background]


def test_high_sparsity_membership() -> None:
    sparse = make_episode(1, 0, ultrasound=(10.0,), markers=(10.0,), index_hours=120.0)
    assert sparsity_of(sparse) > 0.4
    assert sparse in members([sparse], "high_sparsity")


def test_describe_covers_every_stratum() -> None:
    rows = describe([make_episode(i, i % 2) for i in range(6)])
    assert len(rows) == len(STRATA)
