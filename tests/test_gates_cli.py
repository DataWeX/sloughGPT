"""`sloughgpt gates` — the CLI projection of the static gates (card 20261112_060632).

Endpoint rule, Tier-2 external utility: this command surfaces the gate
report to an operator; it never enters the model-facing tool registry.
Pins the projection contract:

1. the subcommand exists and runs to completion on a clean tree (exit 0);
2. output names every gate (an operator must see what was checked);
3. `--strict` additionally fails on *known* (baselined) findings — the
   same semantics Site Doctor's `--strict` has: known debt trips only the
   operator who asked for it;
4. parity with the pytest surface: the CLI exit code agrees with
   `run_gates()` on the same tree (one contract, two projections).

Invocation follows `apps/cli/tests/test_cli_main.py`: `main()` dispatches
off `sys.argv` through the project's own framework (`core.framework`, not
pypi click — CliRunner cannot drive it).
"""

from __future__ import annotations

import pytest

from domain.gates import run_gates


@pytest.fixture
def gates_main(monkeypatch, capsys):
    """Run `sloughgpt gates [args]` via main(); return (exit_code, output)."""

    def _run(*args: str):
        import sys as _sys

        from apps.cli.src import cli as cli_mod

        monkeypatch.setattr(_sys, "argv", ["sloughgpt", "gates", *args])
        code = 0
        try:
            cli_mod.main()
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        return code, capsys.readouterr().out

    return _run


def test_gates_command_exists_and_is_green(gates_main):
    code, out = gates_main()
    assert code == 0, out
    assert "gates" in out.lower()


def test_output_names_every_gate(gates_main):
    _, out = gates_main()
    for report in run_gates():
        assert report.name in out, f"{report.name} missing from CLI output"


def test_strict_fails_when_known_findings_exist(gates_main, monkeypatch):
    """--strict trips on baselined findings; default mode tolerates them."""
    from domain import gates as gates_mod

    seeded = gates_mod.GateReport(name="seeded", ok=True, findings=[], known=["gap"])
    monkeypatch.setattr(gates_mod, "run_gates", lambda *a, **k: [seeded])

    code, out = gates_main()
    assert code == 0, out

    code, out = gates_main("--strict")
    assert code != 0, "--strict must trip on known findings"
    assert "gap" in out


def test_cli_exit_agrees_with_core(gates_main):
    """One contract, two projections: CLI exit code == run_gates() verdict.

    Lenient mode gates on `ok` only — baselined (`known`) debt is tolerated
    exactly as the pytest strict-xfail baseline tolerates it; `--strict`
    is what trips on it (pinned above).
    """
    reports = run_gates()
    all_ok = all(r.ok for r in reports)
    code, _ = gates_main()
    assert (code == 0) == all_ok
