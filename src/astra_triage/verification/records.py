"""Layer one, part B: the claims that need the records, and cannot be run here.

Every check in this module names the cohort it would read and the quantity it would
compute, and every one is reported ``BLOCKED``. None is estimated, borrowed from the
article or filled with an invented value: the derivation cohort, the temporal hold-out,
the controlled trial of the alert and the reading study are held by the participating
institutions, and the external resource is behind a data-use agreement.

Listing them is the point. A release that reports only the checks it can run invites
the reader to assume the rest were run too.
"""

from __future__ import annotations

from astra_triage.verification.checks import Check
from astra_triage.verification.transcription import (
    EXTERNAL_EPISODES,
    TEMPORAL_EPISODES,
)

DERIVATION = "the five-site derivation cohort"
TEMPORAL = f"the held-out temporal cohort ({TEMPORAL_EPISODES} episodes)"
EXTERNAL = f"the credentialed paediatric intensive-care resource ({EXTERNAL_EPISODES} episodes)"
PRIVATE = "the private record of the two prospective windows"
READING = "the reader-level table of the three-arm study"


def checks() -> list[Check]:
    rows = [
        (
            "R01",
            "the derivation event count, site counts and calendar window",
            "Table 1",
            DERIVATION,
        ),
        ("R02", "the temporal cohort counts and its 12.9% event share", "Table 1", TEMPORAL),
        ("R03", "the external cohort counts and its 11.3% event share", "Table 1", EXTERNAL),
        ("R04", "the eight comparison rows of the temporal validation table", "Table 2", TEMPORAL),
        (
            "R05",
            "the paired interval and p-value for ASTRA against late fusion",
            "Table 2 footnote",
            TEMPORAL,
        ),
        ("R06", "the ablation rows and their calibration errors", "Table 3", TEMPORAL),
        (
            "R07",
            "the additive bound computed from the two single-stream scores",
            "Table 3 last row",
            TEMPORAL,
        ),
        ("R08", "the interaction gain and its interval", "Sec. 3.3", TEMPORAL),
        ("R09", "the per-site areas for the three arms", "Table 4 panel A", DERIVATION),
        ("R10", "the leave-one-site-out refits", "Table 4 panel B", DERIVATION),
        ("R11", "the four transported rows and their calibration", "Table 5", EXTERNAL),
        ("R12", "the eight subgroup rows", "Table 6 panel A", TEMPORAL),
        ("R13", "the four sparsity bands and their calibration", "Table 6 panel B", TEMPORAL),
        ("R14", "the three reading arms and both gaps", "Table 7", READING),
        ("R15", "inter-reader agreement in the unaided arm", "Table 7 footnote", READING),
        ("R16", "the eight adjudicated cases and their score trajectories", "Table 8", PRIVATE),
        ("R17", "the two prospective windows", "Table 9", PRIVATE),
        (
            "R18",
            "the threshold fixed at 0.20 before the prospective windows opened",
            "Sec. 3.9",
            PRIVATE,
        ),
        (
            "R19",
            "the calibration slope and intercept of the constrained model",
            "Sec. 3.2",
            TEMPORAL,
        ),
    ]
    return [
        Check(
            identifier,
            "record",
            statement,
            location,
            "BLOCKED",
            note=f"needs {source}",
        )
        for identifier, statement, location, source in rows
    ]
