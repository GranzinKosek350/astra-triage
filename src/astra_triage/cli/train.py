"""Train one arm on a staged episode table."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from torch import nn

from astra_triage.cli.common import (
    apply_drops,
    arm_for,
    cohort_for,
    configure_logging,
    load_experiment,
    model_config,
    split_for,
    train_config,
)
from astra_triage.training.checkpoint import save
from astra_triage.training.loop import History, train_arm
from astra_triage.training.seeding import resolve_device, set_seed

LOGGER = logging.getLogger("astra_triage.train")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="train one arm of the comparison")
    parser.add_argument("--experiment", required=True, help="path to an experiment file")
    parser.add_argument("--cohort", required=True, help="path to the episode table")
    parser.add_argument("--out", default="artifacts/model.pt", help="checkpoint destination")
    parser.add_argument("--seed", type=int, default=None, help="override the run seed")
    parser.add_argument("--steps", type=int, default=None, help="stop after this many batches")
    parser.add_argument("--device", default="auto", help="cpu, cuda, mps or auto")
    parser.add_argument("--override", action="append", default=[], help="key=value override")
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    arguments = build_parser().parse_args(argv)
    experiment = load_experiment(arguments.experiment, arguments.override)
    settings = train_config(experiment)
    seed = settings.seeds[0] if arguments.seed is None else arguments.seed
    set_seed(seed)
    device = resolve_device(arguments.device)
    episodes = apply_drops(experiment, cohort_for(arguments.cohort))
    training = split_for(episodes, "train")
    arm = arm_for(experiment)
    if not isinstance(arm, nn.Module):
        raise SystemExit("a rule arm has no gradient training path; use evaluate --fit-on")
    arm = arm.to(device)
    LOGGER.info(
        "training %s on %d episodes, effective batch %d, %d epochs, %s",
        getattr(arm, "name", type(arm).__name__),
        len(training),
        settings.effective_batch,
        settings.epochs,
        settings.precision,
    )
    closed_form = getattr(arm, "closed_form", None)
    fit = getattr(arm, "fit", None)
    if closed_form and callable(fit):
        fit(training)
        history = History(epochs=[{"total": 0.0}], batches=0)
    else:
        history = train_arm(
            arm,
            training,
            settings,
            device,
            seed=seed,
            capacity=model_config(experiment).capacity,
            steps_limit=arguments.steps,
        )
    target = save(
        Path(arguments.out),
        arm,
        settings,
        seed=seed,
        epoch=len(history.epochs),
        extra={"first_loss": history.first_loss, "last_loss": history.last_loss},
    )
    LOGGER.info("wrote %s after %d batches", target, history.batches)
    for row in history.epochs[-3:]:
        LOGGER.info(
            "epoch total %.4f bce %.4f cal %.4f site %.4f",
            row["total"],
            row["bce"],
            row["calibration"],
            row["site"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
