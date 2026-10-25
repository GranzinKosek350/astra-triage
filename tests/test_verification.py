"""The verification artefacts themselves."""

from __future__ import annotations

import json
from pathlib import Path

from astra_triage.verification import claims, conformance, execution, hygiene, manuscript, records
from astra_triage.verification.checks import Check, counts, overall
from astra_triage.verification.transcription import (
    TABLE2,
    TABLE3,
    TABLE4A,
    TABLE6A,
    TABLE6B,
    TABLE7,
    TABLE9,
)


def test_every_manuscript_check_has_a_status() -> None:
    rows = manuscript.checks()
    assert len(rows) >= 40
    for row in rows:
        assert row.status in {"PASS", "FAIL"}


def test_manuscript_findings_are_reported_rather_than_smoothed() -> None:
    failing = {row.identifier for row in manuscript.checks() if row.status == "FAIL"}
    assert {"M06", "M08", "M15", "M28", "M38", "M41"} <= failing


def test_record_checks_are_blocked_and_named() -> None:
    rows = records.checks()
    assert rows
    assert all(row.status == "BLOCKED" for row in rows)
    assert all(row.note.startswith("needs ") for row in rows)


def test_execution_checks_all_pass() -> None:
    rows = execution.run()
    assert len(rows) >= 27
    failures = [row for row in rows if row.status != "PASS"]
    assert not failures, [row.identifier for row in failures]


def test_conformance_checks_pass() -> None:
    rows = conformance.checks()
    assert all(row.status == "PASS" for row in rows), [
        (row.identifier, row.observed) for row in rows if row.status != "PASS"
    ]


def test_hygiene_checks_pass() -> None:
    rows = hygiene.checks()
    assert all(row.status == "PASS" for row in rows), [
        (row.identifier, row.observed) for row in rows if row.status != "PASS"
    ]


def test_claim_status_is_read_from_checks() -> None:
    rows = manuscript.checks()
    statuses = {row.identifier: row.status for row in rows}
    entries = claims.build(rows)
    blocked = [entry for entry in entries if entry["status"] == "BLOCKED"]
    assert not blocked or all(entry["kind"] == "result" for entry in blocked)
    passing = [entry for entry in entries if entry["status"] == "PASS"]
    assert passing
    for entry in passing:
        assert all(statuses[check] == "PASS" for check in entry["verified_by"])


def test_claim_map_covers_the_article() -> None:
    assert len(claims.CLAIMS) >= 78
    identifiers = [claim.identifier for claim in claims.CLAIMS]
    assert len(identifiers) == len(set(identifiers))


def test_check_rejects_an_unknown_status() -> None:
    try:
        Check("X1", "code", "statement", "location", "OK")
    except ValueError:
        return
    raise AssertionError("an unknown status was accepted")


def test_overall_verdict_is_not_green_with_a_blocked_check() -> None:
    rows = [
        Check("A", "code", "s", "l", "PASS", observed=1),
        Check("B", "record", "s", "l", "BLOCKED"),
    ]
    assert overall(rows) == "PARTIALLY_VERIFIED"
    assert counts(rows)["BLOCKED"] == 1


def test_transcription_tables_close() -> None:
    assert len(TABLE2) == 8
    assert len(TABLE3) == 8
    assert sum(row.episodes for row in TABLE4A) == 3412
    assert sum(row.episodes for row in TABLE6A[:4]) == 3412
    assert len(TABLE6B) == 4
    assert len(TABLE7) == 3
    assert len(TABLE9) == 2


def test_manifest_excludes_itself_and_the_artefact_directory(tmp_path) -> None:
    from astra_triage.verification.report import MANIFEST_NAME, manifest_files, write_manifest

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "module.py").write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts" / "model.pt").write_bytes(b"weights")
    (tmp_path / MANIFEST_NAME).write_text("{}", encoding="utf-8")
    listed = {path.as_posix() for path in manifest_files(tmp_path)}
    assert MANIFEST_NAME not in listed
    assert "src/module.py" in listed
    assert not any(name.startswith("artifacts") for name in listed)
    path = write_manifest(tmp_path, tmp_path / MANIFEST_NAME)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    assert payload["count"] == len(listed)
