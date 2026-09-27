"""cli — ``chargectl`` command line entry point."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .advice import optimize_hint
from .control import clear_limit, probe, set_limit
from .status import BatteryReader

BAR_WIDTH = 24


def _bar(level: int) -> str:
    filled = max(0, min(BAR_WIDTH, round(level * BAR_WIDTH / 100)))
    return "[" + "#" * filled + "-" * (BAR_WIDTH - filled) + "]"


def _fmt_minutes(minutes: int | None) -> str:
    if minutes is None:
        return "—"
    if minutes >= 60:
        return f"{minutes // 60}h{minutes % 60:02d}m"
    return f"{minutes}m"


def _status(reader: BatteryReader) -> int:
    st = reader.read()
    print(f"{st.name}  {_bar(st.level)}  {st.level}%")
    print(
        f"  state     {'charging' if st.is_charging else 'on battery'}"
        f"{' (plugged)' if st.is_plugged and not st.is_charging else ''}"
    )
    print(f"  health    {st.health}")
    if st.voltage_mv > 0:
        print(f"  voltage   {st.voltage_mv} mV")
    if st.current_ma != -1:
        print(f"  current   {st.current_ma} mA")
    if st.is_charging and st.time_to_full_min is not None:
        print(f"  to full   {_fmt_minutes(st.time_to_full_min)}")
    if not st.is_charging and st.time_to_empty_min is not None:
        print(f"  to empty  {_fmt_minutes(st.time_to_empty_min)}")
    if st.simulated:
        print("  source    simulated (no battery device found)")
    return 0


def _advice(reader: BatteryReader) -> int:
    hint = optimize_hint(reader.read())
    print(f"{hint['action']}: {hint['reason']}")
    print(f"suggested charge limit: {hint['limit']}%")
    return 0


def _probe() -> int:
    cap = probe()
    print(f"supported    {cap.supported}")
    print(f"writable     {cap.writable}")
    print(f"path         {cap.path or '—'}")
    print(f"current cap  {cap.current_limit if cap.current_limit is not None else '—'}")
    print(f"reason       {cap.reason}")
    return 0 if cap.supported else 1


def _limit(percent: int | None) -> int:
    result = clear_limit() if percent is None else set_limit(percent)
    state = "ok" if result.applied else "failed"
    print(f"{state}: {result.reason}")
    if result.limit is not None:
        print(f"effective limit: {result.limit}%")
    return 0 if result.applied else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chargectl",
        description="Read battery charge, cap overcharge, and get longevity advice.",
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("status", help="show current charge state")
    sub.add_parser("advice", help="show what to do to maximise battery longevity")
    sub.add_parser("probe", help="report whether charge can be capped on this machine")
    limit = sub.add_parser("limit", help="cap charge at a percent (or 'off')")
    limit.add_argument(
        "percent",
        help="1-100, or 'off' to lift the cap",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reader = BatteryReader()
    if args.command in (None, "status"):
        return _status(reader)
    if args.command == "advice":
        return _advice(reader)
    if args.command == "probe":
        return _probe()
    if args.command == "limit":
        raw = str(args.percent).strip().lower()
        if raw in ("off", "clear", "none", "100"):
            return _limit(None)
        try:
            return _limit(int(raw))
        except ValueError:
            print(f"invalid limit: {args.percent!r} (expected 1-100 or 'off')")
            return 2
    return 0


def entry() -> None:
    sys.exit(main())


if __name__ == "__main__":
    entry()
