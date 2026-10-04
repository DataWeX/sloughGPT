"""CLI for Mole, the always-on monitor: ``python -m domain.core._internal.mole``.

Watch-only, read-only, no AI: one probe sweep per tick (the existing
doctor probes), every tick journaled, a one-line event only when the
findings actually change (dedupe — no alert fatigue). Summary in,
full JSON one flag/file away (per AGENTS "summarize, don't dump").
"""

from __future__ import annotations

import argparse
import json
import sys

from domain.core._internal.mole import default_journal_path, run_watch

_KNOWN_PROBES = ("gates", "benchmarks", "http", "sse", "journey")


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
        description="Mole: probe on a cadence, journal every tick, "
        "alert only on change (no AI, read-only).",
    )
    parser.add_argument(
        "--interval", type=float, default=30.0, metavar="S",
        help="seconds between ticks (default: 30)",
    )
    parser.add_argument(
        "--max-ticks", type=int, default=None, metavar="N",
        help="stop after N ticks (default: run until interrupted)",
    )
    parser.add_argument(
        "--skip", action="append", default=[], choices=_KNOWN_PROBES, metavar="PROBE",
        help=f"skip a probe (one per flag; known: {', '.join(_KNOWN_PROBES)})",
    )
    parser.add_argument(
        "--journal", default=None, metavar="PATH",
        help="journal JSONL path (default: $SLO_MOLE_JOURNAL or "
        "~/.cache/slog-doctor/mole-journal.jsonl)",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="exit non-zero on info findings too (mirrors the doctor flag)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="journal only — suppress the per-change event line",
    )
    parser.add_argument(
        "--json", action="store_true", dest="as_json",
        help="print each change event as one JSON object instead of text",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    journal = args.journal or default_journal_path()

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
            skip=tuple(args.skip),
            journal_path=journal,
            on_event=on_event,
            strict=args.strict,
        )
    except KeyboardInterrupt:  # clean Ctrl-C; journal keeps everything so far
        print("mole: stopped", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
