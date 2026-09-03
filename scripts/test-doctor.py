#!/usr/bin/env python3
"""test-doctor — Fast test failure diagnosis for sloughGPT.

Expo-style: one line per result, expand only on failure.

Usage:
    python scripts/test-doctor.py                              # health scan
    python scripts/test-doctor.py <file>                      # diagnose file
    python scripts/test-doctor.py <file>::TestClass::test_fn  # single test
    python scripts/test-doctor.py --recent                    # recent failures
    python scripts/test-doctor.py --flake                     # flaky detector
    python scripts/test-doctor.py --fix-hint <test>           # suggest fix
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import textwrap
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
HISTORY_FILE = REPO_ROOT / ".test-history.jsonl"

# ── ANSI ───────────────────────────────────────────────────────────────

_NC = os.environ.get("NO_COLOR") or not sys.stdout.isatty()
_b = "" if _NC else "\033[1m"
_d = "" if _NC else "\033[2m"
_r = "" if _NC else "\033[0m"
_rd = "" if _NC else "\033[31m"
_gn = "" if _NC else "\033[32m"
_y = "" if _NC else "\033[33m"
_c = "" if _NC else "\033[36m"
_gr = "" if _NC else "\033[90m"

_ok = f"{_gn}✓{_r}"
_no = f"{_rd}✖{_r}"
_co = f"{_gr}›{_r}"


class Spinner:
    """No-op spinner — just runs the task."""

    def __init__(self, msg: str):
        self.msg = msg

    def start(self) -> "Spinner":
        return self

    def stop(self) -> None:
        pass


def _p(t: str = "", end: str = "\n") -> None:
    sys.stdout.write(t + end)
    sys.stdout.flush()


def _line(icon: str, msg: str) -> None:
    """One line: icon + message."""
    _p(f"  {icon} {msg}")


def _dim(msg: str) -> str:
    return f"{_d}{msg}{_r}"


def _red(msg: str) -> str:
    return f"{_rd}{msg}{_r}"


def _green(msg: str) -> str:
    return f"{_gn}{msg}{_r}"


# ── Pytest ─────────────────────────────────────────────────────────────

def _run(args: list[str], timeout: int = 120, tb: str = "short") -> tuple[int, str, str]:
    cmd = [sys.executable, "-m", "pytest"] + args + [f"--tb={tb}", "-q", "--no-header"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(REPO_ROOT))
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return 1, "", f"TIMEOUT after {timeout}s"


def _counts(stdout: str, stderr: str) -> dict[str, int]:
    c = {"p": 0, "f": 0, "e": 0, "s": 0}
    for line in stdout.splitlines() + stderr.splitlines():
        m = re.search(r"(\d+) passed", line)
        if m: c["p"] = int(m.group(1))
        m = re.search(r"(\d+) failed", line)
        if m: c["f"] = int(m.group(1))
        m = re.search(r"(\d+) error", line)
        if m: c["e"] = int(m.group(1))
        m = re.search(r"(\d+) skipped", line)
        if m: c["s"] = int(m.group(1))
    return c


def _results(stdout: str) -> list[tuple[str, str]]:
    out = []
    for line in stdout.splitlines():
        m = re.match(r"^(PASSED|FAILED|ERROR|SKIPPED)\s+(\S+)", line.strip())
        if m:
            out.append((m.group(2), m.group(1).lower()))
    return out


# ── Diagnosis ──────────────────────────────────────────────────────────

def _source(nodeid: str, n: int = 6) -> str:
    parts = nodeid.split("::")
    if len(parts) < 2:
        return ""
    fp = REPO_ROOT / parts[0]
    if not fp.exists():
        return ""
    try:
        lines = fp.read_text().splitlines()
    except Exception:
        return ""
    name = parts[-1]
    for i, line in enumerate(lines):
        if f"def {name}(" in line or f"def {name}:" in line:
            s, e = max(0, i - n), min(len(lines), i + n + 1)
            return "\n".join(
                f"  {'→' if j == i else ' '} {j+1:4d} │ {lines[j]}"
                for j in range(s, e)
            )
    return ""


def _git_log(filepath: str, n: int = 3) -> list[str]:
    try:
        r = subprocess.run(
            ["git", "log", "--oneline", f"-{n}", "--", filepath],
            capture_output=True, text=True, timeout=5, cwd=str(REPO_ROOT),
        )
        return [l for l in r.stdout.strip().splitlines() if l]
    except Exception:
        return []


def _related(nodeid: str, lim: int = 5) -> list[str]:
    parts = nodeid.split("::")
    if len(parts) < 2:
        return []
    fp = REPO_ROOT / parts[0]
    if not fp.exists():
        return []
    try:
        text = fp.read_text()
    except Exception:
        return []
    results = [
        f"{parts[0]}::{m.group(1)}"
        for m in re.finditer(r"def (test_\w+)\(", text)
        if f"{parts[0]}::{m.group(1)}" != nodeid
    ]
    return results[:lim]


# ── Failure patterns ───────────────────────────────────────────────────

_PAT: list[tuple[str, str, str]] = [
    (r"ImportError: cannot import name '(\w+)' from '(\w+)'", "import", "'{0}' not in '{1}' — check __init__.py exports"),
    (r"ModuleNotFoundError: No module named '(\S+)'", "module", "'{0}' not installed — pip install {0}"),
    (r"AssertionError: (.+)", "assert", "{0}"),
    (r"TypeError: .+takes (\d+) positional.*?but (\d+)", "arity", "expects {0} args, got {1}"),
    (r"KeyError: (.+)", "key", "{0} not found — check spelling"),
    (r"AttributeError: (.+)", "attr", "{0}"),
    (r"FileNotFoundError:.*'(.+?)'", "file", "{0} not found"),
    (r"(asyncio\.)?TimeoutError", "timeout", "timed out — check for deadlock or increase timeout"),
    (r"DID NOT RAISE", "no-raise", "expected exception but none was raised"),
    (r"SystemExit\((\d+)\)", "exit", "sys.exit({0}) called — use exception instead"),
]


def _diagnose(output: str) -> Optional[tuple[str, str]]:
    for pat, label, msg in _PAT:
        m = re.search(pat, output)
        if m:
            try:
                return label, msg.format(*m.groups())
            except (IndexError, KeyError):
                return label, msg
    return None


# ── History ────────────────────────────────────────────────────────────

def _append(results: list[tuple[str, str]], dur: float) -> None:
    entry = {"ts": time.time(), "dur": round(dur, 2), "r": [{"n": n, "o": o} for n, o in results]}
    with open(HISTORY_FILE, "a") as f:
        f.write(json.dumps(entry, separators=(",", ":")) + "\n")


def _history() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    out = []
    with open(HISTORY_FILE) as f:
        for line in f:
            if line.strip():
                try:
                    out.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    pass
    return out


# ═══════════════════════════════════════════════════════════════════════
# Commands — one line per result
# ═══════════════════════════════════════════════════════════════════════

def cmd_health(quiet: bool = False, summary: bool = False, verbose: bool = False) -> int:
    areas = [
        ("CLI framework", "apps/cli/tests/test_slo_cli.py"),
        ("CLI commands", "apps/cli/tests/"),
        ("Core Python", "packages/core-py/tests/"),
        ("Root tests", "tests/"),
    ]

    if not quiet and not summary:
        _p(f"\n  {_b}{_c}Test Health{_r}")

    all_ok = True
    total_p = total_f = total_e = 0

    for name, path in areas:
        full = REPO_ROOT / path
        if not full.exists():
            if not quiet and not summary:
                _line(_co, f"{name:20s} {_dim(path + ' not found')}")
            continue

        sp = Spinner(f"running {name}...").start()
        t0 = time.time()
        code, out, err = _run([path], timeout=120)
        dur = time.time() - t0
        c = _counts(out, err)
        total_p += c["p"]
        total_f += c["f"]
        total_e += c["e"]
        sp.stop()

        if code == 0:
            msg = f"{name:20s} {c['p']} passed  {_dim(f'{dur:.1f}s')}"
            if not quiet and not summary:
                _line(_ok, msg)
        else:
            all_ok = False
            parts = [f'{c["p"]} passed']
            if c["f"]: parts.append(_red(f'{c["f"]} failed'))
            if c["e"]: parts.append(_red(f'{c["e"]} error'))
            msg = f"{name:20s} {', '.join(parts)}  {_dim(f'{dur:.1f}s')}"
            if not quiet and not summary:
                _line(_no, msg)
                if verbose:
                    for line in err.splitlines()[-5:]:
                        _p(f"           {_dim(line.strip())}")

    _p()
    if all_ok:
        _line(_ok, f"{_green('All healthy')}  {_dim(f'{total_p} passed')}")
    else:
        _line(_no, f"{_red(f'{total_f + total_e} failures')}")
        if not quiet and not summary:
            _p(f"           {_dim('run: test-doctor <file> to diagnose')}")
    _p()
    return 0 if all_ok else 1


def cmd_diagnose(target: str) -> int:
    fp = REPO_ROOT / target.split("::")[0]
    if not fp.exists():
        _p(f"\n  {_no} {_red(target + ' not found')}\n")
        return 1

    _p(f"\n  {_b}{_c}Test Doctor{_r}  {target}")

    sp = Spinner("running...").start()
    t0 = time.time()
    code, out, err = _run([target, "--tb=short", "-v"], timeout=60)
    dur = time.time() - t0
    c = _counts(out, err)
    res = _results(out)
    sp.stop()

    parts = []
    if c["p"]: parts.append(_green(f'{c["p"]}P'))
    if c["f"]: parts.append(_red(f'{c["f"]}F'))
    if c["e"]: parts.append(_red(f'{c["e"]}E'))
    if c["s"]: parts.append(f'{c["s"]}S')
    parts.append(_dim(f'{dur:.1f}s'))
    _line(" ", " ".join(parts))

    # Failures — one line each, then expand first
    if code != 0:
        fails = [(n, o) for n, o in res if o in ("failed", "error")]
        for n, _ in fails:
            _line(_no, n)

        # Expand first failure
        if fails:
            first = fails[0][0]
            combined = out + "\n" + err

            # Pattern diagnosis
            diag = _diagnose(combined)
            if diag:
                label, detail = diag
                _p(f"           {_c}{label}{_r}: {detail}")

            # Source
            src = _source(first)
            if src:
                _p()
                for line in src.splitlines():
                    _p(f"  {line}")

            # Git log — one line
            glog = _git_log(first.split("::")[0])
            if glog:
                _p(f"           {_dim(glog[0])}")

            # Related — one line
            rel = _related(first)
            if rel:
                _p(f"           {_dim('also: ' + ', '.join(t.split('::')[-1] for t in rel))}")

    _p(f"\n  {_b}re-run:{_r} python3 -m pytest {target} -v --tb=short\n")

    if res:
        _append(res, dur)

    return code


def cmd_fix_hint(target: str) -> int:
    _p(f"\n  {_b}{_c}Fix Hint{_r}  {target}")

    sp = Spinner("running with full traceback...").start()
    t0 = time.time()
    code, out, err = _run([target, "--tb=long", "-v"], timeout=60)
    dur = time.time() - t0
    sp.stop()

    if code == 0:
        _line(_ok, f"{_green('passing')}  {_dim(f'{dur:.1f}s')}")
        _p()
        return 0

    combined = out + "\n" + err

    # Compact traceback — only the interesting lines
    _p(f"  {_dim('traceback:')}")
    capture = False
    for line in combined.splitlines():
        if "FAILED" in line or "Traceback" in line:
            capture = True
        if capture:
            if line.strip().startswith("===") or line.strip().startswith("---"):
                capture = False
                continue
            # Skip pytest internal frames
            if "site-packages" in line or "_pytest" in line:
                continue
            _p(f"    {line}")

    # Diagnosis
    diag = _diagnose(combined)
    if diag:
        label, detail = diag
        _p(f"\n  {_y}{label}{_r}: {detail}")

    # Source — wider context
    src = _source(target, n=10)
    if src:
        _p()
        for line in src.splitlines():
            _p(f"  {line}")

    # Related
    rel = _related(target)
    if rel:
        _p(f"\n  {_dim('also: ' + ', '.join(t.split('::')[-1] for t in rel))}")

    _p(f"\n  {_b}re-run:{_r} python3 -m pytest {target} -v --tb=long\n")
    return 1


def cmd_recent(n: int = 20) -> int:
    history = _history()
    if not history:
        _p(f"\n  {_dim('no history yet')}\n")
        return 0

    _p(f"\n  {_b}{_c}Recent{_r}  last {n} runs\n")

    recent = history[-n:]
    by_test: dict[str, list[str]] = defaultdict(list)
    for entry in recent:
        ts = time.strftime("%H:%M", time.localtime(entry.get("ts", 0)))
        for r in entry.get("r", []):
            if r.get("o") in ("failed", "error"):
                by_test[r["n"]].append(ts)

    if not by_test:
        _line(_ok, _green("no failures"))
    else:
        for nid, times in sorted(by_test.items(), key=lambda x: -len(x[1])):
            _line(_no, f"{nid}  {_dim(f'{len(times)}x, last {times[-1]}')}")

    _p()
    return 0


def cmd_flake() -> int:
    history = _history()
    _p(f"\n  {_b}{_c}Flaky{_r}\n")

    if len(history) < 2:
        _p(f"  {_dim('need 2+ runs to detect')}\n")
        return 0

    outcomes: dict[str, list[bool]] = defaultdict(list)
    for entry in history:
        seen = set()
        for r in entry.get("r", []):
            nid = r["n"]
            if nid not in seen:
                outcomes[nid].append(r["o"] == "passed")
                seen.add(nid)

    flaky = [
        (nid, sum(r) / len(r), len(r))
        for nid, r in outcomes.items()
        if len(r) >= 2 and any(r) and not all(r)
    ]

    if not flaky:
        _line(_ok, _green("no flakes"))
    else:
        flaky.sort(key=lambda x: x[1])
        for nid, rate, runs in flaky:
            bar_len = 16
            filled = int(rate * bar_len)
            clr = _gn if rate > 0.8 else _y if rate > 0.5 else _rd
            bar = f"{clr}{'█' * filled}{_d}{'░' * (bar_len - filled)}{_r}"
            _p(f"  {_no} {bar} {rate:.0%}  {nid}")

    _p()
    return 0


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(
        prog="test-doctor", description="Fast test failure diagnosis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              test-doctor                              health scan
              test-doctor tests/test_foo.py            diagnose file
              test-doctor --fix-hint <test>::test_fn   suggest fix
        """),
    )
    ap.add_argument("target", nargs="?")
    ap.add_argument("--recent", "-r", action="store_true")
    ap.add_argument("--flake", "-f", action="store_true")
    ap.add_argument("--fix-hint", metavar="TEST")
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--quiet", "-q", action="store_true", help="minimal output")
    ap.add_argument("--summary", "-s", action="store_true", help="summary only")
    ap.add_argument("--verbose", "-v", action="store_true", help="detailed output")
    args = ap.parse_args()

    if args.no_color:
        global _b, _d, _r, _rd, _gn, _y, _c, _gr, _NC
        _NC = True
        _b = _d = _r = _rd = _gn = _y = _c = _gr = ""

    if args.fix_hint: return cmd_fix_hint(args.fix_hint)
    if args.flake: return cmd_flake()
    if args.recent: return cmd_recent()
    if args.target: return cmd_diagnose(args.target)
    return cmd_health(quiet=args.quiet, summary=args.summary)


if __name__ == "__main__":
    sys.exit(main())
