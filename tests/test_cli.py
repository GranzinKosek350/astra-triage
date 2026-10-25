"""The five commands, end to end on a staged table built by the test."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from astra_triage.cli import audit, evaluate, score, train, verify
from astra_triage.cohort.reader import write_episodes
from support import make_episode


@pytest.fixture()
def staged(tmp_path: Path) -> Path:
    rows = []
    sites = ("A", "B", "C")
    for index in range(16):
        episode = make_episode(
            index,
            index % 2,
            site=sites[index % 3],
            age=0.7 + 0.9 * index,
            index_hours=48.0 + 4.0 * index,
            ultrasound=(8.0, 20.0 + index, 34.0),
            markers=(4.0, 20.0 + index, 40.0, 55.0),
            backgrounds=("enterostomy",) if index % 4 == 0 else (),
            vitals=True,
            clinical=True,
        )
        rows.append(
            type(episode)(
                episode_id=episode.episode_id,
                patient_id=episode.patient_id,
                site=episode.site,
                age_years=episode.age_years,
                procedure=episode.procedure,
                index_hours=episode.index_hours,
                ultrasound=episode.ultrasound,
                markers=episode.markers,
                label=episode.label,
                split="train" if index < 12 else "test",
                vitals=episode.vitals,
                clinical=episode.clinical,
                backgrounds=episode.backgrounds,
            )
        )
    return write_episodes(rows, tmp_path / "cohort.jsonl")


def test_train_writes_a_checkpoint(staged: Path, tmp_path: Path, capsys) -> None:
    target = tmp_path / "model.pt"
    code = train.main(
        [
            "--experiment",
            "configs/experiments/_smoke.yaml",
            "--cohort",
            str(staged),
            "--out",
            str(target),
            "--device",
            "cpu",
            "--steps",
            "2",
        ]
    )
    assert code == 0
    assert target.exists()


def test_score_writes_one_line_per_episode(staged: Path, tmp_path: Path) -> None:
    out = tmp_path / "scores.jsonl"
    code = score.main(
        [
            "--experiment",
            "configs/experiments/_smoke.yaml",
            "--cohort",
            str(staged),
            "--split",
            "test",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 4
    assert all(0.0 <= row["score"] <= 1.0 for row in rows)
    assert all(row["band"] in {"low", "borderline", "high"} for row in rows)


def test_evaluate_prints_a_row(staged: Path, capsys) -> None:
    code = evaluate.main(
        [
            "--experiment",
            "configs/experiments/baseline_crp_rule.yaml",
            "--cohort",
            str(staged),
            "--split",
            "test",
            "--resamples",
            "20",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["arm"]["arm"] == "crp_threshold_rule"
    assert payload["arm"]["available"] is True


def test_evaluate_table_scores_every_arm(staged: Path, capsys) -> None:
    code = evaluate.main(
        [
            "--experiment",
            "configs/experiments/_smoke.yaml",
            "--cohort",
            str(staged),
            "--split",
            "test",
            "--table",
            "--resamples",
            "20",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    names = {row["arm"] for row in payload["rows"]}
    assert {"astra", "late_fusion", "tft", "grud", "logistic_last_value"} <= names
    assert payload["sites"]
    assert payload["decision"]


def test_evaluate_reading_study(tmp_path: Path, capsys) -> None:
    payload = {
        "labels": [1, 0, 1, 0, 0, 1],
        "resamples": 20,
        "arms": [
            {"name": "clinician_alone", "probabilities": [[0.6, 0.4, 0.7, 0.2, 0.3, 0.5]] * 2},
            {"name": "model_alone", "probabilities": [[0.8, 0.2, 0.9, 0.1, 0.2, 0.7]] * 2},
            {
                "name": "clinician_with_support",
                "probabilities": [[0.9, 0.1, 0.9, 0.1, 0.1, 0.8]] * 2,
            },
        ],
    }
    path = tmp_path / "reading.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    code = evaluate.main(["--experiment", "configs/experiments/main.yaml", "--reading", str(path)])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert len(out["rows"]) == 3


def test_evaluate_deployment_log(tmp_path: Path, capsys) -> None:
    payload = {
        "threshold_fixed_at": 0.2,
        "silent": {"sites": 3, "months": 6, "episodes": 268, "events": 34},
        "open_label": {"sites": 3, "months": 3, "episodes": 141, "events": 17},
    }
    path = tmp_path / "deployment.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    code = evaluate.main(["--experiment", "configs/experiments/main.yaml", "--deployment", str(path)])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert [row["phase"] for row in out["rows"]] == ["silent", "open_label"]


def test_audit_runs_the_properties(capsys) -> None:
    code = audit.main(["--properties"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert "properties" in payload
    assert payload["properties"]["dominance"]


def test_verify_writes_every_artefact(tmp_path: Path, monkeypatch) -> None:
    root = Path(".").resolve()
    (tmp_path / "configs").symlink_to(root / "configs")
    (tmp_path / "src").symlink_to(root / "src")
    monkeypatch.chdir(tmp_path)
    code = verify.main(["--quiet"])
    assert code == 0
    for name in (
        "claim_to_code.json",
        "verification_report.json",
        "verification_summary.txt",
        "integrity_manifest.json",
    ):
        assert (tmp_path / name).exists()
    report = json.loads((tmp_path / "verification_report.json").read_text(encoding="utf-8"))
    assert report["code_status"] == "PASS"
    assert report["overall_status"] in {"PARTIALLY_VERIFIED", "VERIFIED"}
