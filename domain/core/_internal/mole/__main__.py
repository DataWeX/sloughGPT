"""CLI for Mole — one command, no subcommands.

Default: one read-only sweep over every registered probe → one short
human summary + the report file, then exit (0/1/2 like the report run).
``--watch``: the always-on monitor — probe on a cadence, journal every
tick, one-line event only when the findings change (no AI, read-only).
Summary in, full JSON one flag/file away (per AGENTS "summarize, don't
dump").
"""

from __future__ import annotations

import argparse
import json
import sys

from domain.core._internal.mole import (
    default_journal_path,
    default_report_path,
    run_mole,
    run_watch,
)
from domain.core._internal.mole.models import Finding
from domain.infrastructure._internal.health_flow import Severity

_KNOWN_PROBES = ("gates", "benchmarks", "http", "sse", "journey")
_TOP_FINDINGS = 5


def _summary_lines(report) -> list[str]:
    counts = report.summary["by_severity"]
    lines = [
        f"mole: overall {str(report.overall).upper()} — "
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


def _event_line(line: dict) -> str:
    if "error" in line:
        return f"mole: tick {line['tick']} — WATCH ERROR ({line['error']})"
    counts = line["summary"]["by_severity"]
    return (
        f"mole: tick {line['tick']} — CHANGED overall={line['overall']} "
        f"({counts['critical']} critical, {counts['warn']} warn, "
        f"{counts['info']} info, {counts['ok']} ok)"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m domain.core._internal.mole",
        description="Mole — read-only probe sweep (default: one pass, then the "
        "summary; --watch: cadence + journal, alert only on change). No AI.",
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
        dest="as_json",
        help="print JSON instead of the summary (one pass: the full report; "
        "--watch: one object per change event)",
    )
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        metavar="PROBE",
        help=f"probe to skip — comma-separated (http,sse) or one per flag; "
        f"known: {', '.join(_KNOWN_PROBES)}",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="treat info findings as warn for the exit code",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="keep probing on a cadence, journal every tick (default: one pass)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=30.0,
        metavar="S",
        help="watch: seconds between ticks (default: 30)",
    )
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=None,
        metavar="N",
        help="watch: stop after N ticks (default: run until interrupted)",
    )
    parser.add_argument(
        "--journal",
        default=None,
        metavar="PATH",
        help="watch: journal JSONL path (default: $SLO_MOLE_JOURNAL or "
        "~/.cache/slog-doctor/mole-journal.jsonl)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="watch: journal only — suppress the per-change event line",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    skip = {part.strip() for value in args.skip for part in value.split(",") if part.strip()}
    unknown = skip - set(_KNOWN_PROBES)
    if unknown:
        parser.error(f"unknown probe(s) in --skip: {', '.join(sorted(unknown))}")

    if args.watch:

        def on_event(line: dict, _report) -> None:
            if args.quiet:
                return
            if args.as_json:
                print(json.dumps(line, sort_keys=True), flush=True)
            else:
                print(_event_line(line), flush=True)

        try:
            return run_watch(
                interval_s=args.interval,
                max_ticks=args.max_ticks,
                skip=tuple(sorted(skip)),
                journal_path=args.journal or default_journal_path(),
                on_event=on_event,
                strict=args.strict,
            )
        except KeyboardInterrupt:  # clean Ctrl-C; journal keeps everything so far
            print("mole: stopped", file=sys.stderr)
            return 130

    # Default: one pass over every probe, then show the output.
    report = run_mole(
        run_sweep=not args.no_sweep,
        window_s=args.window,
        skip=skip,
        report_path=args.report,
        write=True,
        preflight="http" not in skip,
    )
    path = args.report or default_report_path()

    if args.as_json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        for line in _summary_lines(report):
            print(line)
        print(f"report: {path}")
    return report.exit_code(strict=args.strict)


if __name__ == "__main__":
    sys.exit(main())
