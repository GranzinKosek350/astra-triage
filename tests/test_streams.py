"""Stream vocabulary, standardisation and the padded batch."""

from __future__ import annotations

import torch

from astra_triage.streams.case import Episode, Observation, _gaps, build_batch, stale_flags
from astra_triage.streams.spec import (
    MARKER_FIELDS,
    ULTRASOUND_FIELDS,
    derive_marker_ratios,
    inverse_transform,
    marker_width,
    static_width,
    transform,
    ultrasound_width,
)
from support import make_episode


def test_widths_match_the_vocabulary() -> None:
    assert ultrasound_width() == len(ULTRASOUND_FIELDS)
    assert marker_width() == len(MARKER_FIELDS) + 3
    assert static_width() == 6


def test_transform_clips_and_standardises() -> None:
    raw = torch.tensor([[0.0, 9.0, -1.0, 1e6, 3.0, 9.0]])
    encoded = transform(ULTRASOUND_FIELDS, raw)
    assert encoded.shape == (1, len(ULTRASOUND_FIELDS))
    assert torch.isfinite(encoded).all()


def test_inverse_transform_recovers_the_units() -> None:
    raw = torch.tensor([[2.5, 1.0, 2.0, 12.0, 1.0, 1.0]])
    encoded = transform(ULTRASOUND_FIELDS, raw)
    recovered = inverse_transform(ULTRASOUND_FIELDS, encoded)
    assert torch.allclose(recovered, raw, atol=1e-5)


def test_derived_ratios_respond_to_the_panel() -> None:
    low = torch.tensor([[9.0, 5.0, 3.0, 250.0, 10.0, 0.2]])
    high = torch.tensor([[15.0, 12.0, 1.0, 250.0, 10.0, 0.2]])
    low_ratio = derive_marker_ratios(transform(MARKER_FIELDS, low))
    high_ratio = derive_marker_ratios(transform(MARKER_FIELDS, high))
    assert high_ratio[0, 0] > low_ratio[0, 0]
    assert high_ratio[0, 2] > low_ratio[0, 2]


def test_gaps_repeat_for_a_repeated_timestamp() -> None:
    assert _gaps([6.0, 18.0, 18.0, 30.0]) == [6.0, 12.0, 12.0, 12.0]


def test_batch_shapes_and_masks(episodes: list[Episode]) -> None:
    batch = build_batch(episodes[:4], capacity=8)
    assert batch.ultrasound.values.shape == (4, 8, ultrasound_width())
    assert batch.markers.values.shape == (4, 8, marker_width())
    assert batch.static.shape == (4, static_width())
    assert batch.label.shape == (4,)
    assert int(batch.ultrasound.mask.sum()) == 12
    assert batch.site.max() < 5


def test_late_observation_is_dropped() -> None:
    episode = make_episode(1, 1, ultrasound=(10.0, 30.0, 70.0), index_hours=40.0)
    visible = episode.visible()
    assert [o.tau_hours for o in visible.ultrasound] == [10.0, 30.0]


def test_absent_stream_is_flagged_and_left_empty() -> None:
    padded = make_episode(1, 1, ultrasound=(), ultrasound_recorded=False)
    batch = build_batch([padded], capacity=4)
    assert int(batch.ultrasound.mask.sum()) == 0
    assert bool(batch.ultrasound.absent[0])


def test_stale_flag_boundary() -> None:
    episode = make_episode(1, 0, ultrasound=(30.0,), index_hours=54.0)
    batch = build_batch([episode], capacity=4)
    flags = stale_flags(batch.ultrasound, batch.index_hours)
    assert not bool(flags[0])
    later = make_episode(2, 0, ultrasound=(29.0,), index_hours=54.0)
    second = build_batch([later], capacity=4)
    assert bool(stale_flags(second.ultrasound, second.index_hours)[0])


def test_observation_rejects_negative_time() -> None:
    try:
        Observation(-1.0, (0.0,) * len(ULTRASOUND_FIELDS))
    except ValueError:
        return
    raise AssertionError("a negative timestamp was accepted")
