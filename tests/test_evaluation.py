"""Arm scoring, sites, subgroups, decision curves, the reading study and the windows."""

from __future__ import annotations

import pytest
import torch

from astra_triage.evaluation.ablation import additive_bound, drop_stream, rank_average
from astra_triage.evaluation.arms import evaluate_arm, paired_delta, scores_of
from astra_triage.evaluation.decision import references, table
from astra_triage.evaluation.deployment import DeploymentPhase, DeploymentRecord
from astra_triage.evaluation.reading import ReadingArm, gaps, summarise
from astra_triage.evaluation.sites import heterogeneity_by_arm, per_site
from astra_triage.evaluation.subgroups import sparsity_table, subgroup_table
from astra_triage.model.astra import build_astra
from astra_triage.model.config import AstraConfig
from support import make_episode


def _config() -> AstraConfig:
    return AstraConfig(
        d_model=16,
        heads=2,
        encoder_hidden=16,
        head_hidden=8,
        gate_hidden=8,
        adapter_rank=4,
        capacity=8,
    )


def _cohort() -> list:
    sites = ("A", "B", "C")
    rows = []
    for index in range(12):
        rows.append(
            make_episode(
                index,
                index % 2,
                site=sites[index % 3],
                age=0.5 + 1.1 * index,
                index_hours=48.0 + 5.0 * index,
                ultrasound=(8.0, 20.0 + index),
                markers=(4.0, 28.0 + index, 50.0),
                backgrounds=("enterostomy",) if index % 3 == 0 else (),
            )
        )
    return rows


def test_evaluate_arm_produces_every_column() -> None:
    model = build_astra(_config())
    rows = _cohort()
    report = evaluate_arm(model, rows, resamples=40)
    payload = report.row()
    assert report.available
    for key in (
        "auroc",
        "auprc",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "brier",
        "ece",
        "slope",
    ):
        assert key in payload
    assert 0.0 <= float(payload["auroc"]) <= 1.0


def test_scores_of_a_rule_arm_reports_unavailability() -> None:
    from astra_triage.comparators.pews import PaediatricEarlyWarning

    scores, reason = scores_of(PaediatricEarlyWarning(), _cohort())
    assert reason is not None
    assert torch.isnan(scores).all()


def test_paired_delta_against_itself_is_zero() -> None:
    rows = _cohort()
    scores = torch.sigmoid(torch.arange(len(rows), dtype=torch.float32))
    labels = torch.tensor([float(e.label) for e in rows])
    difference, low, high, _ = paired_delta(scores, scores, labels, resamples=40)
    assert difference == 0.0
    assert low <= 0.0 <= high


def test_per_site_and_heterogeneity() -> None:
    rows = _cohort()
    scores = torch.sigmoid(torch.arange(len(rows), dtype=torch.float32) / 3.0)
    table_rows = per_site({"astra": scores}, rows)
    assert {row.site for row in table_rows} == {"A", "B", "C"}
    spread = heterogeneity_by_arm({"astra": scores}, rows)
    assert 0.0 <= spread["astra"]["i_squared"] <= 100.0


def test_subgroup_table_skips_groups_without_events() -> None:
    rows = _cohort()
    scores = torch.sigmoid(torch.arange(len(rows), dtype=torch.float32))
    subgroups = subgroup_table({"astra": scores}, rows)
    assert subgroups
    assert all(row.events > 0 for row in subgroups)


def test_sparsity_table_uses_the_four_bands() -> None:
    rows = _cohort()
    scores = torch.sigmoid(torch.arange(len(rows), dtype=torch.float32))
    bands = sparsity_table({"astra": scores}, rows)
    assert all(row.stratum in {"low", "medium", "high", "very_high"} for row in bands)


def test_decision_table_and_references() -> None:
    rows = _cohort()
    labels = torch.tensor([float(e.label) for e in rows])
    arms = {
        "astra": torch.sigmoid(torch.arange(len(rows), dtype=torch.float32)),
        "crp_threshold_rule": torch.full((len(rows),), 0.3),
    }
    rows_out = table(arms, labels)
    assert len(rows_out) == 2
    assert rows_out[0].advantage_over_rule is not None
    refs = references(labels)
    assert refs["treat_none"] == 0.0


def test_reading_summary_and_gaps() -> None:
    rng = torch.Generator().manual_seed(2)
    episodes = 30
    labels = (torch.arange(episodes) % 3 == 0).to(torch.float32)
    alone = torch.rand(4, episodes, generator=rng)
    supported = torch.clamp(alone + 0.15 * labels, max=1.0)
    model = torch.full((4, episodes), 0.4)
    arms = [
        ReadingArm("clinician_alone", alone, 6.8),
        ReadingArm("model_alone", model, 0.4),
        ReadingArm("clinician_with_support", supported, 7.6),
    ]
    rows = summarise(arms, labels)
    assert [row.condition for row in rows] == [
        "clinician_alone",
        "model_alone",
        "clinician_with_support",
    ]
    assert rows[2].auroc >= rows[0].auroc
    values = gaps(arms, labels, resamples=40)
    assert "supported_minus_unaided" in values


def test_deployment_record_table() -> None:
    silent = DeploymentPhase(label="silent", sites=3, months=6, episodes=268, events=34, alerts_per_hundred=28.3)
    opened = DeploymentPhase(label="open_label", sites=3, months=3, episodes=141, events=17, alerts_per_hundred=27.9)
    record = DeploymentRecord(silent=silent, open_label=opened, threshold_fixed_at=0.2)
    rows = record.table()
    assert rows[0]["prevalence"] == pytest.approx(100.0 * 34 / 268)
    assert record.before_after("alerts_per_hundred") == (28.3, 27.9)


def test_drop_stream_removes_the_readings() -> None:
    rows = _cohort()
    without = drop_stream(rows, "ultrasound")
    assert all(not row.ultrasound for row in without)
    assert all(not row.ultrasound_recorded for row in without)


def test_rank_average_is_bounded() -> None:
    first = torch.tensor([0.1, 0.4, 0.35, 0.8])
    second = torch.tensor([0.2, 0.3, 0.6, 0.7])
    blended = rank_average(first, second)
    assert float(blended.min()) == 0.0
    assert float(blended.max()) == 1.0


def test_additive_bound_returns_an_area() -> None:
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
    first = torch.tensor([0.1, 0.9, 0.2, 0.8])
    second = torch.tensor([0.3, 0.7, 0.4, 0.6])
    _, area = additive_bound(first, second, labels)
    assert 0.0 <= area <= 1.0
