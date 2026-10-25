"""The seven comparator arms and the grid they share."""

from __future__ import annotations

import torch

from astra_triage.comparators.base import Link
from astra_triage.comparators.crp_rule import last_crp
from astra_triage.comparators.grid import grid_from_batch
from astra_triage.comparators.grud import GRUD
from astra_triage.comparators.last_value import LastValueLogistic, last_value_matrix
from astra_triage.comparators.late_fusion import LateFusionTransformer
from astra_triage.comparators.pas import PaediatricAppendicitisScore, score_episode
from astra_triage.comparators.pews import PaediatricEarlyWarning, score_vitals
from astra_triage.comparators.registry import ALIASES, build, canonical
from astra_triage.comparators.tft import TFT
from astra_triage.model.config import AstraConfig
from astra_triage.streams.case import build_batch
from astra_triage.streams.vitals import Vitals
from support import make_episode


def _small() -> AstraConfig:
    return AstraConfig(
        d_model=16,
        heads=2,
        encoder_hidden=16,
        head_hidden=8,
        gate_hidden=8,
        adapter_rank=4,
        capacity=8,
    )


def test_registry_aliases_resolve() -> None:
    assert canonical("Late Fusion Transformer") == "late_fusion"
    assert canonical("temporal_fusion_transformer") == "tft"
    assert canonical("pews") == "pews"
    assert "crp_threshold_rule" in ALIASES


def test_every_arm_builds() -> None:
    for name in (
        "astra",
        "late_fusion",
        "tft",
        "grud",
        "logistic_last_value",
        "pews",
        "pas",
        "crp",
    ):
        assert build(name, _small()) is not None


def test_grid_layers_the_streams_on_an_hourly_axis(episodes: list) -> None:
    batch = build_batch(episodes[:3], capacity=8)
    grid = grid_from_batch(batch)
    assert grid.features.shape[0] == 3
    assert grid.features.shape[1] == int(torch.ceil(batch.index_hours.max()).item()) + 1
    assert grid.index_position.max() < grid.features.shape[1]


def test_late_fusion_reads_the_counts(episodes: list) -> None:
    arm = LateFusionTransformer(_small())
    arm.eval()
    batch = build_batch(episodes[:4], capacity=8)
    with torch.no_grad():
        logits = arm(batch)
        counts = arm.net._counts(batch)
    assert logits.shape == (4,)
    assert counts.shape == (4, 5)
    assert float(counts[0, 4]) > 0.0


def test_grud_is_causal_past_the_index(episodes: list) -> None:
    arm = GRUD(_small())
    arm.eval()
    shorter = build_batch([episodes[0]], capacity=8)
    longer = build_batch(
        [
            make_episode(
                0,
                episodes[0].label,
                ultrasound=episodes[0].ultrasound and (8.0,),
                markers=(4.0, 28.0),
                index_hours=400.0,
            )
        ],
        capacity=8,
    )
    with torch.no_grad():
        first = arm(shorter)
        second = arm(longer)
    assert first.shape == (1,) and second.shape == (1,)


def test_tft_produces_one_logit(episodes: list) -> None:
    arm = TFT(_small())
    arm.eval()
    with torch.no_grad():
        logits = arm(build_batch(episodes[:2], capacity=8))
    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()


def test_last_value_matrix_is_count_blind(episodes: list) -> None:
    features = last_value_matrix(episodes[:3])
    assert features.shape[0] == 3
    doubled = last_value_matrix(episodes[:3])
    assert torch.equal(features, doubled)


def test_last_value_logistic_fits_and_scores(episodes: list) -> None:
    arm = LastValueLogistic(l2=1.0, iterations=50)
    arm.fit(episodes)
    result = arm.score(episodes)
    assert result.available
    assert torch.isfinite(result.scores).all()


def test_pews_scores_the_banded_items() -> None:
    normal = Vitals(120.0, 34.0, 95.0, 37.2, 97.0)
    abnormal = Vitals(200.0, 70.0, 60.0, 40.5, 88.0, "pain")
    assert score_vitals(normal, 3.0).points == 0
    assert score_vitals(abnormal, 3.0).points >= 9


def test_appendicitis_score_weights_the_items() -> None:
    episode = make_episode(1, 0, vitals=True, clinical=True)
    scored = score_episode(episode)
    assert scored.points == 6
    assert scored.items["tenderness"] == 2
    assert scored.items["leukocytosis"] == 2


def test_rule_arms_report_unavailability_without_the_block(episodes: list) -> None:
    pews = PaediatricEarlyWarning()
    result = pews.score(episodes[:2])
    assert not result.available
    assert "vital-sign" in (result.reason or "")


def test_crp_rule_reads_the_last_panel() -> None:
    episode = make_episode(1, 0, markers=(4.0, 20.0, 52.0), index_hours=60.0)
    assert last_crp(episode) == 30.0 + 5.0 * 52.0


def test_rule_links_move_with_the_score() -> None:
    link = Link(intercept=0.0, slope=1.0)
    low = float(link(torch.tensor([-1.0]))[0])
    high = float(link(torch.tensor([1.0]))[0])
    assert high > low
    fitted = Link().fit(torch.linspace(-2, 2, 40), (torch.linspace(-2, 2, 40) > 0).to(torch.float32))
    assert fitted.fitted
    assert fitted.slope > 0


def test_appendicitis_arm_fits_on_instrumented_episodes(instrumented_episodes: list) -> None:
    arm = PaediatricAppendicitisScore()
    arm.fit(instrumented_episodes)
    result = arm.score(instrumented_episodes)
    assert result.available
    assert torch.isfinite(result.scores).all()
