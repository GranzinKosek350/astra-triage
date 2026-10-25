"""Run both verification layers and write the four artefacts.

Exit code zero means the code layer passed: every execution, conformance and hygiene
check ran and held. It does not mean the release is fully verified, because the record
layer cannot run without the cohort and the manuscript layer has findings of its own.
Both of those are in the report.
"""

from __future__ import annotations

import argparse
import logging

from astra_triage.verification import report as report_module

LOGGER = logging.getLogger("astra_triage.verify")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="run both verification layers")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    report = report_module.run_all()
    if not arguments.quiet:
        print(
            f"overall {report['overall_status']} | code {report['code_status']} | "
            f"manuscript {report['manuscript_status']} | "
            f"{report['counts']['PASS']} pass, {report['counts']['FAIL']} fail, "
            f"{report['counts']['BLOCKED']} blocked"
        )
    return 0 if report["code_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
