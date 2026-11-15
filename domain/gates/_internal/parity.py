"""Command-parity gate core (the Command Rule in AGENTS.md, mechanically enforced).

Extracted from `tests/core-py/test_command_parity.py` so the product surface
(`sloughgpt gates`, Site Doctor) can run it without pytest.  The test file
imports these helpers and stays the assertion wrapper — same detection,
same messages, same baseline semantics (strict xfail: the known-gap list may
only shrink; a baselined gap that starts passing is reported so the entry
gets deleted).

Detection is deliberately *static* — method source is read, never executed:
`logs --clear` wipes the buffer and `train` starts jobs, so a parity check
must not run a command.  Consequences (documented in the Command Rule):

* only `--long` flags are visible (the rule's vocabulary);
* `--help` is exempt on both sides (implied for every command).
"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

from domain.gates import GateReport

BASELINE_PATH = (
    Path(__file__).resolve().parents[3] / "tests" / "core-py" / "command_parity_baseline.json"
)

_FLAG_DOC = re.compile(r"--[a-z][a-z0-9-]*")
_FLAG_CODE = re.compile(r"""["'](--[a-z][a-z0-9-]*)["']""")
_HELP_FLAG = "--help"


def body(fn) -> str:
    """The method's code with its docstring removed — what actually runs."""
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):  # builtins / unavailable source
        return ""
    parts = src.split('"""')
    return parts[2] if len(parts) >= 3 else src


def declared(fn) -> set[str]:
    """Flags the docstring promises (the declaration)."""
    return set(_FLAG_DOC.findall(fn.__doc__ or "")) - {_HELP_FLAG}


def parsed(fn) -> set[str]:
    """Flags the parser really accepts — quoted literals in the body."""
    return set(_FLAG_CODE.findall(body(fn))) - {_HELP_FLAG}


def rows(name: str) -> set[str]:
    """Flags the option view shows for *name*; empty when it has no view."""
    from domain.shell._internal.option_views import command_rows

    return {str(r["flag"]) for r in command_rows(name) if r.get("flag")} - {_HELP_FLAG}


def sides(name: str, fn) -> tuple[set[str], set[str], set[str]]:
    return declared(fn), parsed(fn), rows(name)


def baseline() -> dict[str, str]:
    if not BASELINE_PATH.exists():
        return {}
    data: dict[str, str] = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    return data


def command_problems(name: str, fn) -> list[str]:
    """The drift list for one command — the exact pytest failure text."""
    declared_f, parsed_f, rows_f = sides(name, fn)
    problems: list[str] = []
    if declared_f - parsed_f:
        problems.append(f"declared but never parsed: {sorted(declared_f - parsed_f)}")
    if parsed_f - declared_f:
        problems.append(f"parsed but never declared: {sorted(parsed_f - declared_f)}")
    if rows_f - declared_f:
        problems.append(f"option rows show undeclared flags: {sorted(rows_f - declared_f)}")
    if rows_f - parsed_f:
        problems.append(f"option rows show unparsed flags: {sorted(rows_f - parsed_f)}")
    return problems


def check() -> GateReport:
    """Run parity over every flag-bearing command; baseline gaps = known."""
    from domain.shell._internal.repl import ShellREPL

    base = baseline()
    findings: list[str] = []
    known: list[str] = []
    for name, fn in sorted(ShellREPL.COMMANDS.items()):
        if not any(sides(name, fn)):
            continue  # no flags anywhere: nothing to compare (not governed)
        problems = command_problems(name, fn)
        if not problems:
            if name in base:
                # strict-xfail mirror: a baselined gap that now passes must
                # be deleted from the baseline, and that deletion is red.
                findings.append(
                    f"{name}: baselined gap now passes — delete its entry "
                    f"from {BASELINE_PATH.name} (strict xfail)"
                )
            continue
        message = f"{name}: " + "; ".join(problems)
        (known if name in base else findings).append(message)
    return GateReport(name="command-parity", ok=not findings, findings=findings, known=known)
