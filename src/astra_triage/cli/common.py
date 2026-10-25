"""Experiment loading shared by the five commands.

A configuration is one YAML file under ``configs/experiments`` that points at the
model, stream, endpoint and optimisation files it composes. Overrides arrive as
``key=value`` pairs with dotted paths, which is the smallest thing that behaves like a
command-line configuration system without adding one.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from astra_triage.cohort.reader import read_episodes
from astra_triage.comparators.registry import Arm, build
from astra_triage.model.config import AstraConfig
from astra_triage.streams.case import Episode
from astra_triage.training.optim import TrainConfig

LOGGER = logging.getLogger("astra_triage")
CONFIG_ROOT: Path = Path("configs")


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(levelname)s %(name)s %(message)s",
    )


def read_yaml(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"no configuration at {source}")
    with source.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{source} does not hold a mapping")
    return payload


def merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _compose(name: str, folder: str) -> dict[str, Any]:
    path = CONFIG_ROOT / folder / f"{name}.yaml"
    return read_yaml(path) if path.exists() else {}


def load_experiment(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load an experiment file with its four component files composed in."""
    experiment = read_yaml(path)
    composed = dict(experiment)
    for key, folder in (
        ("model", "model"),
        ("streams", "streams"),
        ("endpoint", "endpoints"),
        ("optimisation", "optimisation"),
    ):
        value = experiment.get(key)
        if isinstance(value, str):
            composed[key] = _compose(value, folder)
        elif isinstance(value, dict):
            composed[key] = value
    model_overrides = experiment.get("model_overrides")
    if isinstance(model_overrides, dict) and isinstance(composed.get("model"), dict):
        composed["model"] = merge(composed["model"], model_overrides)
    for override in overrides or []:
        _apply_override(composed, override)
    return composed


def _apply_override(payload: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ValueError(f"an override is key=value, received {override}")
    key, raw = override.split("=", 1)
    cursor: dict[str, Any] = payload
    parts = key.split(".")
    for part in parts[:-1]:
        target = cursor.get(part)
        if not isinstance(target, dict):
            raise KeyError(f"override path {key} does not exist")
        cursor = target
    cursor[parts[-1]] = _coerce(raw)


def _coerce(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def model_config(experiment: dict[str, Any]) -> AstraConfig:
    model = experiment.get("model", {})
    defaults = AstraConfig()
    return AstraConfig(
        d_model=int(model.get("d_model", defaults.d_model)),
        heads=int(model.get("heads", defaults.heads)),
        encoder_hidden=int(model.get("encoder_hidden", defaults.encoder_hidden)),
        head_hidden=int(model.get("head_hidden", defaults.head_hidden)),
        gate_hidden=int(model.get("gate_hidden", defaults.gate_hidden)),
        dropout=float(model.get("dropout", defaults.dropout)),
        adapter_rank=int(model.get("adapter_rank", defaults.adapter_rank)),
        capacity=int(model.get("capacity", defaults.capacity)),
        use_time_encoding=bool(model.get("use_time_encoding", defaults.use_time_encoding)),
        use_cross_attention=bool(model.get("use_cross_attention", defaults.use_cross_attention)),
        use_missingness_token=bool(model.get("use_missingness_token", defaults.use_missingness_token)),
        use_site_adapter=bool(model.get("use_site_adapter", defaults.use_site_adapter)),
    )


def train_config(experiment: dict[str, Any]) -> TrainConfig:
    settings = experiment.get("optimisation", {})
    defaults = TrainConfig()
    seeds = experiment.get("seeds", settings.get("seeds", list(defaults.seeds)))
    return TrainConfig(
        epochs=int(experiment.get("epochs", settings.get("epochs", defaults.epochs))),
        batch_size=int(experiment.get("batch_size", settings.get("batch_size", defaults.batch_size))),
        grad_accumulation=int(experiment.get("grad_accumulation", settings.get("grad_accumulation", 1))),
        learning_rate=float(settings.get("learning_rate", defaults.learning_rate)),
        weight_decay=float(settings.get("weight_decay", defaults.weight_decay)),
        warmup_steps=int(settings.get("warmup_steps", defaults.warmup_steps)),
        scheduler=str(settings.get("scheduler", defaults.scheduler)),
        precision=str(settings.get("precision", defaults.precision)),
        grad_clip=float(settings.get("grad_clip", defaults.grad_clip)),
        ema_decay=float(settings.get("ema_decay", defaults.ema_decay)),
        seeds=tuple(int(seed) for seed in seeds),
        world_size=int(experiment.get("world_size", settings.get("world_size", 1))),
        lambda_cal=float(settings.get("lambda_cal", defaults.lambda_cal)),
        lambda_site=float(settings.get("lambda_site", defaults.lambda_site)),
    )


def arm_for(experiment: dict[str, Any]) -> Arm:
    name = str(experiment.get("arm", "astra"))
    LOGGER.debug("building arm %s", name)
    return build(name, model_config(experiment))


def cohort_for(path: str | Path) -> list[Episode]:
    episodes = read_episodes(path)
    LOGGER.info("read %d episodes from %s", len(episodes), Path(path).name)
    return episodes


def apply_drops(experiment: dict[str, Any], episodes: list[Episode]) -> list[Episode]:
    """Remove a stream at the input for the ablation arms of Table 3."""
    from astra_triage.evaluation.ablation import drop_stream

    rows = list(episodes)
    for stream in experiment.get("drop_streams", []) or []:
        rows = drop_stream(rows, str(stream))
        LOGGER.info("dropped the %s stream for %d episodes", stream, len(rows))
    return rows


def split_for(episodes: list[Episode], split: str) -> list[Episode]:
    rows = [e for e in episodes if e.split == split]
    if not rows:
        raise ValueError(f"no episode carries split={split}")
    return rows
