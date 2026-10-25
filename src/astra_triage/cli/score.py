"""Score a staged table without printing a summary.

The command writes one line per episode with the probability, the triage band and the
two staleness flags. It reads nothing after the index time and it applies no threshold
beyond the one in the endpoint file, so a ward can decide what to do with the score.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from astra_triage.cli.common import (
    arm_for,
    cohort_for,
    configure_logging,
    load_experiment,
    model_config,
    split_for,
)
from astra_triage.comparators.base import Comparator
from astra_triage.endpoints.definition import band
from astra_triage.model.astra import Astra
from astra_triage.streams.case import build_batch
from astra_triage.training.checkpoint import load
from astra_triage.training.seeding import resolve_device, set_seed

LOGGER = logging.getLogger("astra_triage.score")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="score episodes at their index time")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default="artifacts/scores.jsonl")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--override", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    arguments = build_parser().parse_args(argv)
    experiment = load_experiment(arguments.experiment, arguments.override)
    set_seed(20260825)
    device = resolve_device(arguments.device)
    episodes = cohort_for(arguments.cohort)
    rows = split_for(episodes, arguments.split)
    arm = arm_for(experiment)
    target = Path(arguments.out)
    target.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(arm, Comparator):
        result = arm.score(rows)
        if not result.available:
            LOGGER.error("the arm cannot read this record: %s", result.reason)
            return 2
        probabilities = result.scores
        stale: dict[str, list[bool]] = {"ultrasound": [], "markers": []}
    else:
        arm = arm.to(device)
        if arguments.checkpoint:
            state = load(arguments.checkpoint, arm)
            LOGGER.info("restored epoch %d at seed %d", state.epoch, state.seed)
        arm.eval()
        with torch.no_grad():
            batch = build_batch(rows, capacity=model_config(experiment).capacity).to(device)
            output = arm(batch)
        if not isinstance(arm, Astra):
            probabilities = torch.sigmoid(output)
            stale = {"ultrasound": [], "markers": []}
        else:
            probabilities = output.probability.cpu()
            stale = {
                "ultrasound": [bool(value) for value in output.ultrasound_stale.cpu()],
                "markers": [bool(value) for value in output.marker_stale.cpu()],
            }

    with target.open("w", encoding="utf-8") as handle:
        for index, episode in enumerate(rows):
            value = float(probabilities[index])
            payload = {
                "episode_id": episode.episode_id,
                "site": episode.site,
                "index_hours": episode.index_hours,
                "score": value,
                "band": band(value).value,
                "ultrasound_stale": (bool(stale["ultrasound"][index]) if stale["ultrasound"] else None),
                "markers_stale": bool(stale["markers"][index]) if stale["markers"] else None,
            }
            handle.write(json.dumps(payload) + "\n")
    LOGGER.info("wrote %d scores to %s", len(rows), target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
