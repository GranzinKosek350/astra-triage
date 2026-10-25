"""Forward pass, count-blindness and the missingness mechanisms."""

from __future__ import annotations

import pytest
import torch

from astra_triage.constraints import count_blind
from astra_triage.model.astra import Astra, build_astra
from astra_triage.model.config import AstraConfig
from astra_triage.streams.case import build_batch
from support import make_episode


def test_forward_returns_one_probability(small_config: AstraConfig, episodes: list) -> None:
    model = build_astra(small_config)
    batch = build_batch(episodes[:5], capacity=small_config.capacity)
    output = model(batch)
    assert output.probability.shape == (5,)
    assert bool((output.probability >= 0).all() and (output.probability <= 1).all())
    assert output.representation.shape == (5, small_config.d_model)
    assert output.site_logits.shape == (5, small_config.site_count)


def test_replication_leaves_the_score_unchanged(small_config: AstraConfig, episodes: list) -> None:
    model = build_astra(small_config)
    result = count_blind.audit(model, episodes[:6], factor=3)
    assert result.replication < 1e-5
    assert result.permutation < 1e-5


def test_replication_changes_a_counting_model(episodes: list) -> None:
    class Counting(torch.nn.Module):
        def forward(self, batch: object) -> torch.Tensor:
            live = batch.ultrasound.mask.sum(dim=1) + batch.markers.mask.sum(dim=1)
            return torch.sigmoid(live.to(torch.float32) - 5.0)

    deviation = count_blind.replication_deviation(Counting(), episodes[:4], factor=2)
    assert deviation > 1e-3


def test_absent_stream_still_scores(small_config: AstraConfig) -> None:
    model = build_astra(small_config)
    absent = [make_episode(index, index % 2, ultrasound=(), ultrasound_recorded=False) for index in range(3)]
    present = [make_episode(index, index % 2, ultrasound=(10.0, 30.0)) for index in range(3)]
    with torch.no_grad():
        first = model(build_batch(absent, capacity=small_config.capacity)).probability
        second = model(build_batch(present, capacity=small_config.capacity)).probability
    assert torch.isfinite(first).all()
    assert float((first - second).abs().max()) > 1e-6


def test_missingness_token_fires_after_a_day(small_config: AstraConfig) -> None:
    model = build_astra(small_config)
    model.ultrasound_encoder.tokens.stale.data.fill_(0.75)
    fresh = build_batch([make_episode(1, 0, ultrasound=(30.0,), index_hours=54.0)], capacity=8)
    stale = build_batch([make_episode(1, 0, ultrasound=(29.0,), index_hours=54.0)], capacity=8)
    with torch.no_grad():
        first = model(fresh)
        second = model(stale)
    assert not bool(first.ultrasound_stale[0])
    assert bool(second.ultrasound_stale[0])
    assert abs(float(first.probability[0]) - float(second.probability[0])) > 1e-7


def test_stale_token_is_off_when_disabled() -> None:
    config = AstraConfig(d_model=16, heads=2, encoder_hidden=16, adapter_rank=4, use_missingness_token=False)
    model = build_astra(config)
    batch = build_batch([make_episode(1, 0, ultrasound=(29.0,), index_hours=54.0)], capacity=8)
    output = model(batch)
    assert bool(output.ultrasound_stale[0])
    assert float(model.ultrasound_encoder.tokens.stale.detach().abs().sum()) == 0.0


def test_ablations_change_the_parameter_count(small_config: AstraConfig) -> None:
    full = build_astra(small_config).parameter_count()
    without_adapter = build_astra(small_config.variant(use_site_adapter=False)).parameter_count()
    without_time = build_astra(small_config.variant(use_time_encoding=False)).parameter_count()
    assert without_adapter < full
    assert without_time < full


def test_identity_of_fusion_when_disabled(small_config: AstraConfig) -> None:
    config = small_config.variant(use_cross_attention=False)
    model = build_astra(config)
    batch = build_batch([make_episode(2, 1)], capacity=8)
    output = model(batch)
    assert torch.isfinite(output.probability).all()


def test_eval_mode_is_deterministic(small_config: AstraConfig, episodes: list) -> None:
    model = build_astra(small_config)
    model.eval()
    batch = build_batch(episodes[:4], capacity=small_config.capacity)
    with torch.no_grad():
        first = model(batch).probability
        second = model(batch).probability
    assert torch.equal(first, second)


def test_parameter_count_is_stable(small_config: AstraConfig) -> None:
    model: Astra = build_astra(small_config)
    assert model.parameter_count() > 5000
    assert model.parameter_count() == sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_bad_widths_are_rejected() -> None:
    with pytest.raises(ValueError):
        AstraConfig(d_model=30, heads=4)
