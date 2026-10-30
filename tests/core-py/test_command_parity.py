"""The command-parity gate — the Command Rule (AGENTS.md), mechanically enforced.

A command's flag vocabulary appears in three places: its docstring (the
declaration), its parser (the behaviour) and — when it has an option view —
the row registry shown by `options <command>`.  The rule says those three
agree; this test is that rule.

Coverage: every command registered in ``ShellREPL.COMMANDS`` (the explicit
specs plus the auto-registered LinuxCommandsMixin set).  A command with no
flags on *any* side has nothing to compare, so it is not parametrized at
all — only flag-bearing commands can fail (10 today).

Known documentation gaps ride as **strict xfails** loaded from
``command_parity_baseline.json``: the list may only shrink.  When someone
fixes a gap the node starts passing, strict-xfail turns that into a
failure (XPASS), and the entry must be deleted from the JSON.

Detection is deliberately *static* — the method source is read, never
executed: `logs --clear` wipes the buffer and `train` starts jobs, so a
parity check must not run a command.  Two consequences worth knowing:

* only ``--long`` flags are visible (the rule's vocabulary); a command
  using a bare short flag with no long form is outside the gate;
* ``--help`` is exempt on both sides — the rule declares it implied for
  every command, so it is never "declared" and never drifts.
"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import pytest

from domain.shell._internal.repl import ShellREPL

_BASELINE_PATH = Path(__file__).with_name("command_parity_baseline.json")

_FLAG_DOC = re.compile(r"--[a-z][a-z0-9-]*")
_FLAG_CODE = re.compile(r"""["'](--[a-z][a-z0-9-]*)["']""")
_HELP_FLAG = "--help"


def _body(fn) -> str:
    """The method's code with its docstring removed — what actually runs."""
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):  # builtins / unavailable source
        return ""
    parts = src.split('"""')
    return parts[2] if len(parts) >= 3 else src


def _declared(fn) -> set[str]:
    """Flags the docstring promises (the declaration)."""
    return set(_FLAG_DOC.findall(fn.__doc__ or "")) - {_HELP_FLAG}


def _parsed(fn) -> set[str]:
    """Flags the parser really accepts — quoted literals in the body."""
    return set(_FLAG_CODE.findall(_body(fn))) - {_HELP_FLAG}


def _rows(name: str) -> set[str]:
    """Flags the option view shows for *name*; empty when it has no view."""
    from domain.shell._internal.option_views import command_rows

    return {str(r["flag"]) for r in command_rows(name) if r.get("flag")} - {_HELP_FLAG}


def _sides(name: str, fn) -> tuple[set[str], set[str], set[str]]:
    return _declared(fn), _parsed(fn), _rows(name)


def _baseline() -> dict[str, str]:
    if not _BASELINE_PATH.exists():
        return {}
    return json.loads(_BASELINE_PATH.read_text(encoding="utf-8"))


def _governed() -> list:
    """Parametrize over flag-bearing commands; baseline entries carry strict xfail."""
    baseline = _baseline()
    params = []
    for name, fn in sorted(ShellREPL.COMMANDS.items()):
        declared, parsed, rows = _sides(name, fn)
        if not (declared or parsed or rows):
            continue
        marks = []
        if name in baseline:
            marks.append(
                pytest.mark.xfail(strict=True, reason=f"baselined drift: {baseline[name]}")
            )
        params.append(pytest.param(name, marks=marks, id=name))
    return params


def test_detector_reads_a_real_command():
    """Self-check: if the detector ever stops seeing flags, this fails first."""
    fn = ShellREPL.COMMANDS["logs"]
    assert "--level" in _parsed(fn) and "--stats" in _parsed(fn)
    assert "--level" in _declared(fn)
    assert not _parsed(fn) - _declared(fn), "logs is the reference: doc == parser"
    # the option view shows the same vocabulary card D pinned by hand
    assert _rows("logs") <= _declared(fn)
    # a command with no flags anywhere is invisible to the gate
    bare = ShellREPL.COMMANDS["pwd"]
    assert not _declared(bare) and not _parsed(bare) and not _rows("pwd")


@pytest.mark.parametrize("command", _governed())
def test_flag_parity(command: str):
    """declared == parsed, and any option rows are a subset of both."""
    fn = ShellREPL.COMMANDS[command]
    declared, parsed, rows = _sides(command, fn)
    problems: list[str] = []
    if declared - parsed:
        problems.append(f"declared but never parsed: {sorted(declared - parsed)}")
    if parsed - declared:
        problems.append(f"parsed but never declared: {sorted(parsed - declared)}")
    if rows - declared:
        problems.append(f"option rows show undeclared flags: {sorted(rows - declared)}")
    if rows - parsed:
        problems.append(f"option rows show unparsed flags: {sorted(rows - parsed)}")
    if problems:
        pytest.fail(
            f"{command}: "
            + "; ".join(problems)
            + "  (one declaration — see the Command Rule in AGENTS.md; "
            + f"baselined gaps live in {_BASELINE_PATH.name})"
        )
