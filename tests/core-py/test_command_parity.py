"""The command-parity gate — the Command Rule (AGENTS.md), mechanically enforced.

A command's flag vocabulary appears in three places: its docstring (the
declaration), its parser (the behaviour) and — when it has an option view —
the row registry shown by `options <command>`.  The rule says those three
agree; this test is that rule.

As of card 20261112_060632 the *detection* lives in
`domain/gates/_internal/parity.py` (product code, so `sloughgpt gates` and
Site Doctor can run it without the test runner); this file is the thin
assertion wrapper over that core — same detection, same messages, same
strict-xfail baseline semantics.  One contract, two projections.

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

import pytest

from domain.gates._internal import parity
from domain.shell._internal.repl import ShellREPL

_BASELINE_PATH = parity.BASELINE_PATH
_baseline = parity.baseline
_declared = parity.declared
_parsed = parity.parsed
_rows = parity.rows
_sides = parity.sides


def _governed() -> list:
    """Parametrize over flag-bearing commands; baseline entries carry strict xfail."""
    baseline = _baseline()
    params = []
    for name, fn in sorted(ShellREPL.COMMANDS.items()):
        if not any(_sides(name, fn)):
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
    problems = parity.command_problems(command, fn)
    if problems:
        pytest.fail(
            f"{command}: "
            + "; ".join(problems)
            + "  (one declaration — see the Command Rule in AGENTS.md; "
            + f"baselined gaps live in {_BASELINE_PATH.name})"
        )
