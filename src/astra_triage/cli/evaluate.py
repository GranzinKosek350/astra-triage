"""Score a partition, a full table, a reading study or a deployment log.

Four modes, one command, because all four answer the same question: what would the
article's table say on the records I have staged.

``--table`` scores every arm in the registry on the same episodes and adds the per-site,
subgroup, sparsity and decision-curve views. ``--reading`` reads a reader-by-episode
matrix. ``--deployment`` reads the two prospective windows. Without either flag the
command scores the single arm named in the experiment file.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch

from astra_triage.cli.common import (
    apply_drops,
    arm_for,
    cohort_for,
    configure_logging,
    load_experiment,
    model_config,
    split_for,
)
from astra_triage.comparators.base import Comparator
from astra_triage.comparators.registry import LEARNED_ARMS, RULE_ARMS, build
from astra_triage.evaluation import decision, deployment, reading, sites, subgroups
from astra_triage.evaluation.arms import evaluate_arm, scores_of, unavailable_report
from astra_triage.training.checkpoint import load
from astra_triage.training.seeding import resolve_device, set_seed
from astra_triage.verification.report import json_safe

LOGGER = logging.getLogger("astra_triage.evaluate")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="evaluate an arm or a whole table")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--cohort", default=None, help="episode table for scoring")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--checkpoint-dir",
        default="artifacts",
        help="directory holding one <arm>.pt per learned arm, used by --table",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--name", default=None, help="label for the single-arm row")
    parser.add_argument("--fit-on", default="train", help="partition used to fit rule links")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--table", action="store_true", help="score every arm in the registry")
    parser.add_argument("--reading", default=None, help="reader matrix as JSON")
    parser.add_argument("--deployment", default=None, help="deployment log as JSON")
    parser.add_argument("--json", default=None, help="write the payload to this file")
    parser.add_argument("--override", action="append", default=[])
    return parser


def _single(arguments: argparse.Namespace, experiment: dict[str, Any]) -> dict[str, Any]:
    if arguments.cohort is None:
        raise SystemExit("--cohort is required unless --reading or --deployment is given")
    episodes = apply_drops(experiment, cohort_for(arguments.cohort))
    arm = arm_for(experiment)
    if isinstance(arm, Comparator):
        arm.fit(split_for(episodes, arguments.fit_on))
    else:
        arm = arm.to(resolve_device(arguments.device))
        if arguments.checkpoint:
            state = load(arguments.checkpoint, arm)
            LOGGER.info("restored epoch %d at seed %d", state.epoch, state.seed)
    rows = split_for(episodes, arguments.split)
    report = evaluate_arm(arm, rows, name=arguments.name, resamples=arguments.resamples)
    if not report.available:
        LOGGER.warning("the arm could not read the record: %s", report.reason)
    return {"arm": report.row()}


def _table(arguments: argparse.Namespace, experiment: dict[str, Any]) -> dict[str, Any]:
    if arguments.cohort is None:
        raise SystemExit("--table needs --cohort")
    episodes = apply_drops(experiment, cohort_for(arguments.cohort))
    training = split_for(episodes, arguments.fit_on)
    scoring = split_for(episodes, arguments.split)
    labels = torch.tensor([float(e.label) for e in scoring], dtype=torch.float32)
    settings = model_config(experiment)
    scored: dict[str, torch.Tensor] = {}
    rows: list[dict[str, Any]] = []
    for name in LEARNED_ARMS + RULE_ARMS:
        arm = build(name, settings)
        if isinstance(arm, Comparator):
            arm.fit(training)
        else:
            arm = arm.to(resolve_device(arguments.device))
            source = Path(arguments.checkpoint_dir) / f"{name}.pt"
            if name == "astra" and arguments.checkpoint:
                source = Path(arguments.checkpoint)
            if not source.exists():
                rows.append(unavailable_report(name, len(scoring), f"no checkpoint at {source}").row())
                continue
            load(source, arm)
        report = evaluate_arm(arm, scoring, name=name, resamples=arguments.resamples)
        rows.append(report.row())
        values, reason = scores_of(arm, scoring)
        if reason is None:
            scored[name] = values
    return {
        "rows": rows,
        "sites": [
            {"site": row.site, "episodes": row.episodes, "events": row.events, "scores": row.scores}
            for row in sites.per_site(scored, scoring)
        ],
        "subgroups": [
            {
                "subgroup": row.subgroup,
                "episodes": row.episodes,
                "events": row.events,
                "scores": row.scores,
            }
            for row in subgroups.subgroup_table(scored, scoring)
        ],
        "sparsity": [
            {
                "stratum": row.stratum,
                "episodes": row.episodes,
                "events": row.events,
                "auroc": row.auroc,
                "ece": row.ece,
                "slope": row.slope,
            }
            for row in subgroups.sparsity_table(scored, scoring)
        ],
        "decision": [
            {
                "arm": row.arm,
                "threshold": row.threshold,
                "net_benefit": row.net_benefit,
                "advantage_over_rule": row.advantage_over_rule,
            }
            for row in decision.table(scored, labels, rule="crp")
        ],
        "decision_references": decision.references(labels),
    }


def _reading(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    labels = torch.tensor(payload["labels"], dtype=torch.float32)
    arms = [
        reading.ReadingArm(
            name=entry["name"],
            probabilities=torch.tensor(entry["probabilities"], dtype=torch.float32),
            median_minutes=entry.get("median_minutes"),
        )
        for entry in payload["arms"]
    ]
    rows = reading.summarise(arms, labels)
    gaps = reading.gaps(arms, labels, resamples=int(payload.get("resamples", 2000)))
    return {
        "rows": [
            {
                "condition": row.condition,
                "auroc": row.auroc,
                "sensitivity": row.sensitivity,
                "specificity": row.specificity,
                "median_minutes": row.median_minutes,
            }
            for row in rows
        ],
        "gaps": {
            name: {"difference": value[0], "low": value[1], "high": value[2], "p": value[3]}
            for name, value in gaps.items()
        },
    }


def _deployment(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    phases = {}
    for key in ("silent", "open_label"):
        entry = payload[key]
        phases[key] = deployment.DeploymentPhase(
            label=key,
            sites=int(entry["sites"]),
            months=int(entry["months"]),
            episodes=int(entry["episodes"]),
            events=int(entry["events"]),
            latency_within_five_minutes=entry.get("latency_within_five_minutes"),
            alerts_per_hundred=entry.get("alerts_per_hundred"),
            reviewed_same_shift=entry.get("reviewed_same_shift"),
            alert_to_review_minutes=entry.get("alert_to_review_minutes"),
            reading_minutes_per_alert=entry.get("reading_minutes_per_alert"),
            time_to_reintervention_hours=entry.get("time_to_reintervention_hours"),
            delayed_beyond_48h=entry.get("delayed_beyond_48h"),
            exploration_without_infection=entry.get("exploration_without_infection"),
        )
    record = deployment.DeploymentRecord(
        silent=phases["silent"],
        open_label=phases["open_label"],
        threshold_fixed_at=float(payload["threshold_fixed_at"]),
    )
    return {"rows": record.table()}


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    arguments = build_parser().parse_args(argv)
    experiment = load_experiment(arguments.experiment, arguments.override)
    set_seed(20260825)
    if arguments.reading:
        payload = _reading(arguments.reading)
    elif arguments.deployment:
        payload = _deployment(arguments.deployment)
    elif arguments.table:
        payload = _table(arguments, experiment)
    else:
        payload = _single(arguments, experiment)
    text = json.dumps(json_safe(payload), indent=2, allow_nan=False)
    print(text)
    if arguments.json:
        Path(arguments.json).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
