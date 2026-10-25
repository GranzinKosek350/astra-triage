"""Shared inputs: episode objects built for the checks, and a small architecture."""

from __future__ import annotations

import pytest
import torch

from astra_triage.model.config import AstraConfig
from support import make_episode


@pytest.fixture()
def small_config() -> AstraConfig:
    return AstraConfig(
        d_model=32,
        heads=2,
        encoder_hidden=32,
        head_hidden=16,
        gate_hidden=16,
        adapter_rank=4,
        capacity=16,
    )


@pytest.fixture()
def episodes() -> list:
    torch.manual_seed(0)
    sites = ("A", "B", "C", "D", "E")
    rows = []
    for index in range(12):
        rows.append(
            make_episode(
                index,
                index % 2,
                site=sites[index % 5],
                age=0.8 + 0.9 * index,
                index_hours=48.0 + 4.0 * index,
                ultrasound=(8.0, 20.0 + index, 34.0),
                markers=(3.0, 15.0, 27.0 + index, 41.0),
                backgrounds=("necrotising_enterocolitis",) if index % 4 == 0 else (),
            )
        )
    return rows


@pytest.fixture()
def instrumented_episodes() -> list:
    return [make_episode(index, index % 2, vitals=True, clinical=True) for index in range(8)]
