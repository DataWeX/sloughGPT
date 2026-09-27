"""cli — ``chargectl`` command line entry point."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .advice import optimize_hint
from .control import clear_limit, probe, set_limit
from .daemon import run_daemon
from .policy import default_policy_path, explain, load_policy, normalize, save_policy
from .status import BatteryReader, default_sys_base

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


def _probe(sys_base: str | None = None) -> int:
    cap = probe(sys_base or default_sys_base())
    print(f"supported    {cap.supported}")
    print(f"writable     {cap.writable}")
    print(f"path         {cap.path or '—'}")
    print(f"current cap  {cap.current_limit if cap.current_limit is not None else '—'}")
    print(f"floor node   {cap.start_supported} ({cap.start_path or '—'})")
    print(f"floor now    {cap.current_floor if cap.current_floor is not None else '—'}")
    print(f"incumbent    {cap.incumbent or 'none detected'}")
    print(f"reason       {cap.reason}")
    return 0 if cap.supported else 1


def _policy_show() -> int:
    policy, error = load_policy()
    cap = probe(default_sys_base())
    print(f"policy file  {default_policy_path()}")
    print(f"enabled      {policy.enabled}")
    print(f"band         {policy.band}%  (mode: {policy.mode})")
    print(f"interval     {policy.interval_seconds:g}s")
    print(f"capability   {explain(policy, cap)}")
    if cap.current_limit is not None:
        print(f"cap in force {cap.current_limit}%")
    if cap.current_floor is not None:
        print(f"floor in force {cap.current_floor}%")
    if error:
        print(f"policy file unreadable ({error}) — showing defaults")
        return 1
    return 0


def _policy_set(args: argparse.Namespace) -> int:
    base, error = load_policy()
    if error:
        print(f"policy file unreadable ({error}) — starting from defaults")
    enabled: bool | None = None
    if args.enable:
        enabled = True
    elif args.disable:
        enabled = False
    policy, err = normalize(
        floor=args.floor,
        ceiling=args.ceiling,
        mode=args.mode,
        enabled=enabled,
        interval_seconds=args.interval,
        base=base,
    )
    if err:
        print(f"invalid policy: {err}")
        return 2
    path = save_policy(policy)
    print(f"saved {path}")
    print(
        f"{'enabled' if policy.enabled else 'disabled'} · band {policy.band}% · mode {policy.mode}"
    )
    print(explain(policy, probe(default_sys_base())))
    return 0


def _daemon(args: argparse.Namespace) -> int:
    return run_daemon(interval=args.interval, once=args.once, sys_base=default_sys_base())


def _limit(percent: int | None, sys_base: str | None = None) -> int:
    base = sys_base or default_sys_base()
    result = clear_limit(sys_base=base) if percent is None else set_limit(percent, sys_base=base)
    state = "ok" if result.applied else "failed"
    print(f"{state}: {result.reason}")
    if result.limit is not None:
        print(f"effective limit: {result.limit}%")
    return 0 if result.applied else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chargectl",
        description="Read battery charge, cap overcharge, and manage a longevity band.",
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
    policy = sub.add_parser("policy", help="show or change the charge policy")
    policy_sub = policy.add_subparsers(dest="policy_command")
    policy_sub.add_parser("show", help="show the current policy (default)")
    policy_sub.add_parser("enable", help="turn enforcement on")
    policy_sub.add_parser("disable", help="turn enforcement off")
    pset = policy_sub.add_parser("set", help="set the band, mode, or interval")
    pset.add_argument("--floor", type=int, default=None, help="resume charging at this percent")
    pset.add_argument("--ceiling", type=int, default=None, help="stop charging at this percent")
    pset.add_argument("--mode", choices=("band", "ceiling"), default=None)
    pset.add_argument("--interval", type=float, default=None, help="seconds between ticks")
    pset.add_argument("--enable", action="store_true", help="enable enforcement too")
    pset.add_argument("--disable", action="store_true", help="disable enforcement too")
    daemon = sub.add_parser("daemon", help="enforce the policy on an interval")
    daemon.add_argument("--once", action="store_true", help="run one tick and exit")
    daemon.add_argument("--interval", type=float, default=None, help="seconds between ticks")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reader = BatteryReader(sys_base=default_sys_base())
    if args.command in (None, "status"):
        return _status(reader)
    if args.command == "advice":
        return _advice(reader)
    if args.command == "probe":
        return _probe()
    if args.command == "daemon":
        return _daemon(args)
    if args.command == "policy":
        command = args.policy_command or "show"
        if command == "show":
            return _policy_show()
        if command in ("enable", "disable"):
            args = argparse.Namespace(
                floor=None,
                ceiling=None,
                mode=None,
                interval=None,
                enable=command == "enable",
                disable=command == "disable",
            )
        return _policy_set(args)
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
