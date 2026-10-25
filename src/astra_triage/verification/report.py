"""Assembling the three artefacts.

The report separates the two layers and keeps them apart in the verdict. ``code_status``
answers whether the released modules run and reproduce independently computed answers.
``manuscript_status`` answers whether the numbers printed in the article close on
themselves. ``overall_status`` is the release-level statement and is
``PARTIALLY_VERIFIED`` whenever any check is blocked, failed or not run, which is the
honest verdict for a release whose cohort is not distributed.

The report contains no timestamp, no duration and no host detail: two runs of the
verification driver on the same tree must produce the same bytes, otherwise the
manifest that digests the tree could never agree with itself.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch

from astra_triage.verification import claims as claims_module
from astra_triage.verification import conformance, execution, hygiene, manuscript, records
from astra_triage.verification.checks import Check, counts, overall

RELEASE_ROOT: Path = Path(".")
MANIFEST_NAME: str = "integrity_manifest.json"
MANIFEST_EXCLUDED: tuple[str, ...] = (
    "artifacts",
    "__pycache__",
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
)


def environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "implementation": platform.python_implementation(),
        "container": {
            "dockerfile": "present",
            "build": "NOT_RUN",
            "reason": "the build host carries no container daemon; the image was not built here",
        },
    }


def _status_of(checks: list[Check]) -> str:
    if any(check.status == "FAIL" for check in checks):
        return "FAIL"
    if any(check.status in {"BLOCKED", "NOT_RUN"} for check in checks):
        return "PARTIALLY_VERIFIED"
    return "PASS"


def collect() -> tuple[list[Check], list[Check], list[Check], list[Check], list[Check]]:
    manuscript_checks = manuscript.checks()
    record_checks = records.checks()
    conformance_checks = conformance.checks()
    execution_checks = execution.run()
    hygiene_checks = hygiene.checks()
    return manuscript_checks, record_checks, conformance_checks, execution_checks, hygiene_checks


def build(
    manuscript_checks: list[Check],
    record_checks: list[Check],
    conformance_checks: list[Check],
    execution_checks: list[Check],
    hygiene_checks: list[Check],
) -> dict[str, Any]:
    everything = manuscript_checks + record_checks + conformance_checks + execution_checks + hygiene_checks
    code = execution_checks + conformance_checks + hygiene_checks
    claim_entries = claims_module.build(everything)
    claim_counts = {"PASS": 0, "FAIL": 0, "BLOCKED": 0, "NOT_RUN": 0}
    for entry in claim_entries:
        claim_counts[entry["status"]] += 1
    return {
        "release": {
            "package": "astra_triage",
            "version": "1.0.0",
            "article": (
                "Temporal Fusion of Postoperative Ultrasound and Inflammatory Markers "
                "for Intra-Abdominal Infection Triage"
            ),
            "venue": "Frontiers",
        },
        "overall_status": overall(everything),
        "code_status": _status_of(code),
        "manuscript_status": _status_of(manuscript_checks),
        "environment": environment(),
        "counts": counts(everything),
        "verification_layer_1": {
            "what": (
                "each article statement checked against its implementation and, where "
                "possible, against the article's own arithmetic"
            ),
            "claims_total": len(claim_entries),
            "claims_by_status": claim_counts,
            "claims_untested": claims_module.uncovered(claim_entries),
            "manuscript_checks": [check.as_dict() for check in manuscript_checks],
            "record_checks": [check.as_dict() for check in record_checks],
            "conformance_checks": [check.as_dict() for check in conformance_checks],
        },
        "verification_layer_2": {
            "what": (
                "the released modules executed on tensors built inside the check process, "
                "compared with independently computed answers"
            ),
            "execution_checks": [check.as_dict() for check in execution_checks],
            "summary": execution.summary(execution_checks),
        },
        "hygiene": {
            "checks": [check.as_dict() for check in hygiene_checks],
            "scanned_files": hygiene.summary().files,
            "source_modules": hygiene.summary().source_files,
        },
        "reproduce": {
            "verify": "bash scripts/verify.sh",
            "tests": "pytest -q",
            "train": "bash scripts/train.sh configs/experiments/main.yaml <cohort.jsonl>",
            "evaluate": "bash scripts/evaluate.sh configs/experiments/main.yaml <cohort.jsonl>",
        },
    }


def write_claims(entries: list[dict[str, Any]], path: Path) -> Path:
    payload = {
        "release": "astra_triage",
        "about": (
            "one row per statement the article makes; the status is read from the checks "
            "named in verified_by, never written by hand"
        ),
        "total": len(entries),
        "claims": entries,
    }
    return _write_json(path, payload)


def json_safe(value: Any) -> Any:
    """Map a non-finite float to null, so the artefact is valid JSON."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, sort_keys=False, allow_nan=False)
        handle.write("\n")
    return path


def write_report(report: dict[str, Any], path: Path) -> Path:
    return _write_json(path, report)


def write_summary(report: dict[str, Any], path: Path) -> Path:
    lines = [
        "ASTRA triage release - verification summary",
        "",
        f"overall status      : {report['overall_status']}",
        f"code status         : {report['code_status']}",
        f"manuscript status   : {report['manuscript_status']}",
        f"checks              : {report['counts']['PASS']} pass, {report['counts']['FAIL']} fail, "
        f"{report['counts']['BLOCKED']} blocked, {report['counts']['NOT_RUN']} not run",
        f"claims              : {report['verification_layer_1']['claims_total']} mapped, "
        f"{report['verification_layer_1']['claims_by_status']['PASS']} pass, "
        f"{report['verification_layer_1']['claims_by_status']['FAIL']} fail, "
        f"{report['verification_layer_1']['claims_by_status']['BLOCKED']} blocked, "
        f"{report['verification_layer_1']['claims_by_status']['NOT_RUN']} not run",
        "",
        "Manuscript checks",
    ]
    for check in report["verification_layer_1"]["manuscript_checks"]:
        lines.append(f"  {check['id']} {check['status']:7s} {check['statement']}")
    lines.append("")
    lines.append("Execution checks")
    for check in report["verification_layer_2"]["execution_checks"]:
        lines.append(f"  {check['id']} {check['status']:7s} {check['statement']}")
    lines.append("")
    lines.append("Record checks (blocked: the cohort is not distributed)")
    for check in report["verification_layer_1"]["record_checks"]:
        lines.append(f"  {check['id']} {check['status']:7s} {check['note']}: {check['statement']}")
    lines.append("")
    lines.append("Hygiene")
    for check in report["hygiene"]["checks"]:
        lines.append(f"  {check['id']} {check['status']:7s} {check['statement']}")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _digest(path: Path) -> str:
    handle = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 16), b""):
            handle.update(block)
    return handle.hexdigest()


def manifest_files(root: Path = RELEASE_ROOT) -> list[Path]:
    rows: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if path.name == MANIFEST_NAME:
            continue
        if any(part in MANIFEST_EXCLUDED for part in relative.parts):
            continue
        rows.append(relative)
    return rows


def write_manifest(root: Path = RELEASE_ROOT, path: Path | None = None) -> Path:
    target = path if path is not None else root / MANIFEST_NAME
    entries = {relative.as_posix(): _digest(root / relative) for relative in manifest_files(root)}
    payload = {
        "algorithm": "sha256",
        "about": (
            "one digest per shipped file, written after every other artefact; the manifest " "does not digest itself"
        ),
        "count": len(entries),
        "files": entries,
    }
    return _write_json(target, payload)


def run_all() -> dict[str, Any]:
    manuscript_checks, record_checks, conformance_checks, execution_checks, hygiene_checks = collect()
    report = build(manuscript_checks, record_checks, conformance_checks, execution_checks, hygiene_checks)
    all_checks = manuscript_checks + record_checks + conformance_checks + execution_checks + hygiene_checks
    write_claims(claims_module.build(all_checks), Path("claim_to_code.json"))
    write_report(report, Path("verification_report.json"))
    write_summary(report, Path("verification_summary.txt"))
    write_manifest(Path("."), Path(MANIFEST_NAME))
    return report
