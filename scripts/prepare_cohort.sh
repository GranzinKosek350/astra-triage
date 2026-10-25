#!/usr/bin/env bash
# Check that a staged episode table is readable and that its partitions are disjoint.
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-$PWD/src}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
COHORT="${1:?usage: prepare_cohort.sh <cohort.jsonl>}"

"$PYTHON_BIN" - "$COHORT" <<'PY'
import sys
from astra_triage.cohort.reader import read_episodes
from astra_triage.cohort.split import assert_patient_disjoint, event_count, site_distribution

episodes = read_episodes(sys.argv[1])
assert_patient_disjoint(episodes)
events, share = event_count(episodes)
print(f"episodes {len(episodes)} | events {events} ({share:.1f}%) | sites {site_distribution(episodes)}")
for split in sorted({e.split for e in episodes}):
    rows = [e for e in episodes if e.split == split]
    found, rate = event_count(rows)
    print(f"  {split:12s} episodes {len(rows):5d} events {found:4d} ({rate:.1f}%)")
PY
