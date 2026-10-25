#!/usr/bin/env bash
# Score a partition and print the row. Pass --table for every arm at once.
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-$PWD/src}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
EXPERIMENT="${1:?usage: evaluate.sh <experiment.yaml> <cohort.jsonl> [split] [checkpoint]}"
COHORT="${2:?usage: evaluate.sh <experiment.yaml> <cohort.jsonl> [split] [checkpoint]}"
SPLIT="${3:-test}"
CHECKPOINT="${4:-}"
ARGS=(--experiment "$EXPERIMENT" --cohort "$COHORT" --split "$SPLIT")
if [[ -n "$CHECKPOINT" ]]; then ARGS+=(--checkpoint "$CHECKPOINT"); fi
"$PYTHON_BIN" -m astra_triage.cli.evaluate "${ARGS[@]}"
