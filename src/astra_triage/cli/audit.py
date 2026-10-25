"""The admissibility audit and the three appendix properties.

Two commands in one entry point. With a cohort and a checkpoint it measures the two
constraints of the admissible set on the fitted model: the replication and permutation
probes for count-blindness, and a fresh site probe for position. Without a cohort it
runs the three properties of the appendix on the observation process, which need no
records.
"""

from __future__ import annotations

import argparse
import json
import logging

import torch

from astra_triage.cli.common import (
    arm_for,
    cohort_for,
    configure_logging,
    load_experiment,
    split_for,
)
from astra_triage.constraints.admissibility import assess
from astra_triage.model.astra import Astra
from astra_triage.properties.counting_estimator import consistency_sweep
from astra_triage.properties.dominance import dominance
from astra_triage.properties.locf_bias import slope_sweep
from astra_triage.training.checkpoint import load
from astra_triage.training.seeding import set_seed
from astra_triage.verification.report import json_safe

LOGGER = logging.getLogger("astra_triage.audit")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="audit the admissible set")
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--cohort", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--properties", action="store_true", help="run the appendix properties")
    parser.add_argument("--out", default=None)
    return parser


def _properties() -> dict[str, object]:
    return {
        "last_observation": [
            {
                "slope": result.slope,
                "gap_coefficient": result.gap_coefficient,
                "carried_area": result.carried_area,
                "carried_and_gap_area": result.carried_and_gap_area,
                "area_gain": result.area_gain,
            }
            for result in slope_sweep()
        ],
        "counting_estimator": [
            {
                "slope": result.slope,
                "population_risk": result.population_risk,
                "cohort_risk": result.cohort_risk,
                "bias": result.bias,
                "within_cohort_ece": result.within_cohort_ece,
            }
            for result in consistency_sweep()
        ],
        "dominance": [
            {
                "slope": slope,
                "admissible_risk": dominance(slope).admissible_risk,
                "unconstrained_risk": dominance(slope).unconstrained_risk,
                "earned": dominance(slope).earned,
                "nested": dominance(slope).nested,
            }
            for slope in (0.0, 1.0, 2.0)
        ],
    }


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    arguments = build_parser().parse_args(argv)
    set_seed(20260825)
    payload: dict[str, object] = {}
    if arguments.properties:
        payload["properties"] = _properties()
    if arguments.experiment and arguments.cohort:
        experiment = load_experiment(arguments.experiment)
        arm = arm_for(experiment)
        if not isinstance(arm, Astra):
            LOGGER.error(
                "the audit applies to the fused arm; %s has no merged representation",
                type(arm).__name__,
            )
            return 2
        if arguments.checkpoint:
            state = load(arguments.checkpoint, arm)
            LOGGER.info("restored epoch %d at seed %d", state.epoch, state.seed)
        episodes = split_for(cohort_for(arguments.cohort), arguments.split)
        with torch.no_grad():
            report = assess(arm, episodes)
        payload["admissibility"] = report.as_dict()
    if not payload:
        LOGGER.error("nothing to audit: pass --properties or a cohort with an experiment")
        return 2
    text = json.dumps(json_safe(payload), indent=2, allow_nan=False)
    print(text)
    if arguments.out:
        with open(arguments.out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
