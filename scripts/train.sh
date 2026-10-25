#!/usr/bin/env bash
# Train one arm. The episode table is private and must be staged by the caller.
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-$PWD/src}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
EXPERIMENT="${1:?usage: train.sh <experiment.yaml> <cohort.jsonl> [out.pt]}"
COHORT="${2:?usage: train.sh <experiment.yaml> <cohort.jsonl> [out.pt]}"
OUT="${3:-artifacts/model.pt}"
"$PYTHON_BIN" -m astra_triage.cli.train --experiment "$EXPERIMENT" --cohort "$COHORT" --out "$OUT"
