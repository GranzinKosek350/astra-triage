#!/usr/bin/env bash
# Run both verification layers and write the four artefacts. Exits non-zero when the
# code layer fails; the record layer is reported as blocked, never as passed.
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-$PWD/src}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -m astra_triage.cli.verify
