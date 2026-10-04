"""CLI for the site doctor: ``python -m domain.core._internal.doctor``.

Phase A is report-only: read-only probes, one JSON report, one short
human summary (full JSON one flag/file away — per AGENTS "summarize,
don't dump").
"""

from __future__ import annotations

import argparse
import json
import sys

from domain.core._internal.doctor import run_doctor
from domain.core._internal.doctor.models import Finding
from domain.core._internal.doctor.report import default_report_path
from domain.infrastructure._internal.health_flow import Severity

_KNOWN_PROBES = ("gates", "benchmarks", "http", "sse", "journey")
_TOP_FINDINGS = 5


def _summary_lines(report) -> list[str]:
    counts = report.summary["by_severity"]
    lines = [
        f"site-doctor: overall {str(report.overall).upper()} — "
        f"{counts['critical']} critical, {counts['warn']} warn, "
        f"{counts['info']} info, {counts['ok']} ok ({report.summary['total']} findings)",
    ]
    if report.probes:
        rendered = ", ".join(
            f"{p['name']}={'skipped' if p['ok'] is None else ('ok' if p['ok'] else 'FAILED')}"
            for p in report.probes
        )
        lines.append(f"probes: {rendered}")
    interesting: list[Finding] = [f for f in report.findings if f.severity != Severity.OK]
    for finding in interesting[:_TOP_FINDINGS]:
        lines.append(
            f"  [{str(finding.severity).upper():8s}] {finding.source}/{finding.check}: "
            f"{finding.message}"
        )
    if len(interesting) > _TOP_FINDINGS:
        lines.append(f"  … {len(interesting) - _TOP_FINDINGS} more finding(s) in the report file")
    if not interesting:
        lines.append("  no issues — all checks nominal")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m domain.core._internal.doctor",
        description="Site doctor — read-only probe sweep over the live stack (Phase A).",
    )
    parser.add_argument(
        "--no-sweep",
        action="store_true",
        help="skip the browser journey sweep; read the existing journey report",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=9.0,
        metavar="N",
        help="SSE observation window in seconds (default: 9)",
    )
    parser.add_argument(
        "--report",
        metavar="PATH",
        default=None,
        help="report path (default: $SLO_DOCTOR_REPORT or ~/.cache/slog-doctor/findings-report.json)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the full report JSON to stdout instead of the summary",
    )
    parser.add_argument(
        "--skip",
        metavar="LIST",
        default="",
        help="comma-separated probes to skip: sse,http,journey",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="treat info findings as warn for the exit code",
    )
    args = parser.parse_args(argv)

    skip = {part.strip() for part in args.skip.split(",") if part.strip()}
    unknown = skip - set(_KNOWN_PROBES)
    if unknown:
        parser.error(f"unknown probe(s) in --skip: {', '.join(sorted(unknown))}")

    report = run_doctor(
        run_sweep=not args.no_sweep,
        window_s=args.window,
        skip=skip,
        report_path=args.report,
        write=True,
        preflight="http" not in skip,
    )
    path = args.report or default_report_path()

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        for line in _summary_lines(report):
            print(line)
        print(f"report: {path}")

    return report.exit_code(strict=args.strict)


if __name__ == "__main__":
    sys.exit(main())
