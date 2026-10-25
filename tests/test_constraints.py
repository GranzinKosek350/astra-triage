"""The admissibility audit on a small model."""

from __future__ import annotations

import torch

from astra_triage.constraints.admissibility import SITE_PROBE_LIFT_LIMIT, assess
from astra_triage.constraints.count_blind import audit, permute, replicate
from astra_triage.constraints.site_probe import batch_sites, probe, representations
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


def test_replicate_triples_the_readings() -> None:
    episode = make_episode(1, 0, ultrasound=(8.0, 20.0), markers=(4.0, 28.0))
    doubled = replicate([episode], factor=2)[0]
    assert len(doubled.ultrasound) == 4
    assert len(doubled.markers) == 4


def test_permute_keeps_every_reading() -> None:
    episode = make_episode(1, 0, ultrasound=(8.0, 20.0, 30.0), markers=(4.0, 28.0, 40.0))
    shuffled = permute([episode], seed=3)[0]
    assert sorted(o.tau_hours for o in shuffled.ultrasound) == [8.0, 20.0, 30.0]


def test_count_blind_audit_passes_on_the_fused_model() -> None:
    model = build_astra(_config())
    result = audit(model, [make_episode(i, i % 2) for i in range(5)], factor=2)
    assert result.passes
    assert result.replication < 1e-5


def test_site_labels_come_from_the_episode() -> None:
    rows = [make_episode(i, 0, site="C") for i in range(3)]
    assert torch.equal(batch_sites(rows), torch.full((3,), 2, dtype=torch.long))


def test_representations_have_the_model_width() -> None:
    model = build_astra(_config())
    features, sites = representations(model, [make_episode(i, i % 2) for i in range(4)])
    assert features.shape == (4, 16)
    assert sites.shape == (4,)


def test_site_probe_returns_a_lift() -> None:
    model = build_astra(_config())
    rows = [make_episode(i, i % 2, site="AB"[i % 2]) for i in range(8)]
    result = probe(model, rows[:4], rows[4:], epochs=20)
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.5 <= result.majority_accuracy <= 1.0


def test_admissibility_report_is_serialisable() -> None:
    model = build_astra(_config())
    rows = [make_episode(i, i % 2, site="ABC"[i % 3]) for i in range(6)]
    report = assess(model, rows)
    payload = report.as_dict()
    assert set(payload) == {
        "replication_deviation",
        "permutation_deviation",
        "count_blind",
        "site_probe_accuracy",
        "site_majority_accuracy",
        "site_lift",
        "site_lift_limit",
        "site_invariant",
        "admissible",
    }
    assert payload["site_lift_limit"] == SITE_PROBE_LIFT_LIMIT
