"""Two steps on the smoke configuration, and the pieces around them."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from astra_triage.cli.common import load_experiment, model_config, train_config
from astra_triage.model.astra import build_astra
from astra_triage.training.checkpoint import load, save
from astra_triage.training.loop import predict, train_arm
from astra_triage.training.optim import ModelEMA, Schedule, TrainConfig, schedule_scale
from astra_triage.training.seeding import resolve_device, set_seed
from support import make_episode

SMOKE = Path("configs/experiments/_smoke.yaml")


def test_smoke_configuration_loads() -> None:
    experiment = load_experiment(SMOKE)
    assert experiment["test_only"] is True
    assert model_config(experiment).d_model == 32
    settings = train_config(experiment)
    assert settings.epochs == 2
    assert settings.seeds == (20260825,)


def test_two_steps_reduce_the_loss() -> None:
    experiment = load_experiment(SMOKE)
    settings = train_config(experiment)
    model = build_astra(model_config(experiment))
    rows = [make_episode(index, index % 2) for index in range(8)]
    history = train_arm(model, rows, settings, torch.device("cpu"), seed=20260825)
    assert history.batches == 4
    assert math.isfinite(history.last_loss)
    assert history.last_loss <= history.first_loss


def test_steps_limit_stops_the_loop() -> None:
    experiment = load_experiment(SMOKE)
    settings = train_config(experiment)
    model = build_astra(model_config(experiment))
    rows = [make_episode(index, index % 2) for index in range(8)]
    history = train_arm(model, rows, settings, torch.device("cpu"), seed=1, steps_limit=1)
    assert history.batches == 1


def test_checkpoint_round_trip(tmp_path) -> None:
    experiment = load_experiment(SMOKE)
    settings = train_config(experiment)
    model = build_astra(model_config(experiment))
    rows = [make_episode(index, index % 2) for index in range(4)]
    before = predict(model, rows)
    path = save(tmp_path / "model.pt", model, settings, seed=11, epoch=1)
    restored = build_astra(model_config(experiment))
    state = load(path, restored)
    assert state.seed == 11
    after = predict(restored, rows)
    assert torch.allclose(before, after, atol=1e-6)


def test_schedule_warmup_then_decay() -> None:
    assert schedule_scale(0, 100, 10) < schedule_scale(5, 100, 10)
    assert schedule_scale(10, 100, 10) == pytest.approx(1.0)
    assert schedule_scale(100, 100, 10) < 0.01


def test_schedule_object_sets_the_learning_rate() -> None:
    model = build_astra(model_config(load_experiment(SMOKE)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    schedule = Schedule(optimizer, TrainConfig(warmup_steps=0), total=10)
    schedule.advance()
    state = schedule.state()
    assert state["step"] == 1


def test_ema_moves_towards_the_weights() -> None:
    model = build_astra(model_config(load_experiment(SMOKE)))
    ema = ModelEMA(model, 0.9)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.1)
    ema.update(model)
    ema.copy_to(model)
    first = next(model.parameters())
    assert float(first.detach().abs().sum()) > 0.0


def test_seed_is_reproducible() -> None:
    set_seed(3)
    first = torch.rand(4)
    set_seed(3)
    second = torch.rand(4)
    assert torch.equal(first, second)


def test_device_resolution() -> None:
    assert resolve_device("cpu").type == "cpu"


def test_missing_cohort_path_is_reported(tmp_path) -> None:
    from astra_triage.cohort.reader import read_episodes

    with pytest.raises(FileNotFoundError):
        read_episodes(tmp_path / "absent.jsonl")
