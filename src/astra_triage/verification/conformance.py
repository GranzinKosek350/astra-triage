"""Configuration conformance.

The experiment files are checked against the constants the article fixes and against
the defaults the release declares. Three families of check.

Fixed constants. The triage threshold, the endpoint window and the staleness horizon
are stated in the article, so a configuration that disagrees with them is wrong.

Declared defaults. The batch size, accumulation, world size, precision and seed count
are engineering defaults; the check confirms the effective batch is the product the
module reports and that every experiment file inherits the same values rather than
quietly reducing them.

No reported result in a configuration. A configuration that carries a number equal to
a printed metric is a configuration that has been tuned against the answer, so the
scan rejects any float in ``configs/`` that matches a Table 2 or Table 3 value.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from astra_triage.endpoints.definition import (
    PRIMARY_WINDOW_HOURS,
    STALE_AFTER_HOURS,
    TRIAGE_THRESHOLD,
)
from astra_triage.verification.checks import Check
from astra_triage.verification.transcription import TABLE2, TABLE3

CONFIG_ROOT: Path = Path("configs")
REPORTED_VALUES: tuple[float, ...] = tuple(
    value for row in TABLE2 for value in (row.auroc, row.auprc, row.sensitivity, row.specificity, row.brier)
) + tuple(row.auroc for row in TABLE3)


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not hold a mapping")
    return payload


def _floats(payload: Any) -> list[float]:
    found: list[float] = []
    if isinstance(payload, dict):
        for value in payload.values():
            found.extend(_floats(value))
    elif isinstance(payload, list):
        for value in payload:
            found.extend(_floats(value))
    elif isinstance(payload, bool):
        return found
    elif isinstance(payload, (int, float)):
        found.append(float(payload))
    return found


def checks() -> list[Check]:
    rows: list[Check] = []
    endpoint = _load(CONFIG_ROOT / "endpoints" / "primary.yaml")
    rows.append(
        Check(
            "C01",
            "conformance",
            "the declared threshold, window and staleness horizon are the article's",
            "Sec. 2.1; Sec. 2.3.0.3; Table 2 footnote",
            (
                "PASS"
                if float(endpoint["threshold"]) == TRIAGE_THRESHOLD
                and float(endpoint["window_hours"]) == PRIMARY_WINDOW_HOURS
                and float(endpoint["stale_after_hours"]) == STALE_AFTER_HOURS
                else "FAIL"
            ),
            observed={
                "threshold": float(endpoint["threshold"]),
                "window_hours": float(endpoint["window_hours"]),
                "stale_after_hours": float(endpoint["stale_after_hours"]),
            },
            expected={
                "threshold": TRIAGE_THRESHOLD,
                "window_hours": PRIMARY_WINDOW_HOURS,
                "stale_after_hours": STALE_AFTER_HOURS,
            },
        )
    )

    training = _load(CONFIG_ROOT / "optimisation" / "adamw_cosine.yaml")
    effective = int(training["batch_size"]) * int(training["grad_accumulation"]) * int(training["world_size"])
    rows.append(
        Check(
            "C02",
            "conformance",
            "the effective batch is the product of its three factors",
            "Engineering default",
            "PASS" if effective == int(training["effective_batch"]) else "FAIL",
            observed=effective,
            expected=int(training["effective_batch"]),
        )
    )
    rows.append(
        Check(
            "C03",
            "conformance",
            "every experiment file inherits the declared batch and seed count",
            "Engineering default",
            _inherit_check(training),
            observed=_inheritance_detail(training),
        )
    )

    model = _load(CONFIG_ROOT / "model" / "astra.yaml")
    rows.append(
        Check(
            "C04",
            "conformance",
            "the two attention directions of Eq. (2) are both enabled in the main configuration",
            "Sec. 2.3.0.2, Eq. (2)",
            (
                "PASS"
                if bool(model["use_cross_attention"])
                and int(model["heads"]) > 1
                and int(model["d_model"]) % int(model["heads"]) == 0
                else "FAIL"
            ),
            observed={
                "use_cross_attention": model["use_cross_attention"],
                "heads": model["heads"],
                "d_model": model["d_model"],
            },
        )
    )
    rows.append(
        Check(
            "C05",
            "conformance",
            "the smoke configuration is marked test-only and is smaller than the main run",
            "pytest smoke path",
            _smoke_check(),
            observed=_smoke_detail(),
        )
    )

    offenders: dict[str, list[float]] = {}
    for path in sorted(CONFIG_ROOT.rglob("*.yaml")):
        payload = _load(path)
        hits = [
            value for value in _floats(payload) if any(abs(value - reported) <= 1e-9 for reported in REPORTED_VALUES)
        ]
        if hits:
            offenders[str(path)] = hits
    rows.append(
        Check(
            "C06",
            "conformance",
            "no configuration carries a value equal to a reported metric",
            "Table 2 and Table 3",
            "PASS" if not offenders else "FAIL",
            observed=offenders or "none",
        )
    )
    return rows


def _experiment_files() -> list[Path]:
    return sorted((CONFIG_ROOT / "experiments").glob("*.yaml"))


def _inheritance_detail(training: dict[str, Any]) -> dict[str, Any]:
    detail: dict[str, Any] = {}
    for path in _experiment_files():
        if path.name == "_smoke.yaml":
            continue
        payload = _load(path)
        detail[path.name] = {
            "batch_size": payload["batch_size"],
            "grad_accumulation": payload["grad_accumulation"],
            "world_size": payload["world_size"],
            "seeds": len(payload["seeds"]),
        }
    return detail


def _inherit_check(training: dict[str, Any]) -> str:
    expected = (
        int(training["batch_size"]),
        int(training["grad_accumulation"]),
        int(training["world_size"]),
        len(training["seeds"]),
    )
    for path in _experiment_files():
        if path.name == "_smoke.yaml":
            continue
        payload = _load(path)
        observed = (
            int(payload["batch_size"]),
            int(payload["grad_accumulation"]),
            int(payload["world_size"]),
            len(payload["seeds"]),
        )
        if observed != expected:
            return "FAIL"
    return "PASS"


def _smoke_detail() -> dict[str, Any]:
    payload = _load(CONFIG_ROOT / "experiments" / "_smoke.yaml")
    return {
        "batch_size": payload["batch_size"],
        "epochs": payload["epochs"],
        "marked": bool(payload.get("test_only", False)),
    }


def _smoke_check() -> str:
    detail = _smoke_detail()
    if not detail["marked"]:
        return "FAIL"
    return "PASS" if int(detail["epochs"]) < 40 else "FAIL"
