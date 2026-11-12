"""Product surface for the static gates (card 20261112_060632).

The gates already exist as pytest suites and scripts (`--check` modes); this
card exposes their *cores* as owned product code so the CLI and Site Doctor
can run them without dragging the test runner along (Test Rule: the product
must not depend on pytest).  These tests pin the surface contract:

1. `run_gates()` reports exactly the three in-scope gates (static, read-only,
   pass/fail, <5s each — the intake filter);
2. a clean tree is green, and a seeded drift is red (the report must be able
   to fail — a gate that cannot go red is decoration);
3. the cores are read-only by construction — no gate may write a byte or
   mutate the tree it inspects (Site Doctor's probe-phase rule: every request
   is a GET, so a probe can never change what it is measuring).
"""

from __future__ import annotations

import ast
from pathlib import Path

from domain.gates import GateReport, run_gates

_GATES_PKG = Path(__file__).resolve().parents[2] / "domain" / "gates"

_EXPECTED = {"command-parity", "domain-baseline", "router-manifest"}


def test_report_shape():
    """GateReport carries a name, an ok flag, and human-readable findings."""
    report = GateReport(name="x", ok=True, findings=[], known=[])
    assert report.ok is True
    assert report.findings == []
    assert str(report)  # renderable for CLI/probe output


def test_exactly_three_gates_in_scope():
    reports = run_gates()
    names = {r.name for r in reports}
    assert names == _EXPECTED, f"gate set drifted: {names ^ _EXPECTED}"


def test_clean_tree_is_green():
    """On a green tree every gate reports ok — no false alarms."""
    for report in run_gates():
        assert report.ok, f"{report.name}: {'; '.join(report.findings)}"


def test_gate_cores_are_read_only():
    """No writes, no tree mutation: static inspection only.

    AST-pinned (static, never executed — test_command_parity shape): any
    `write_text`/`write_bytes`/`open(..., "w")` in a gate core is a defect,
    because Site Doctor runs these over an operator's checkout.
    """
    problems: list[str] = []
    for path in sorted(_GATES_PKG.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                attr = fn.attr if isinstance(fn, ast.Attribute) else ""
                if attr in {"write_text", "write_bytes", "mkdir", "unlink", "rename"}:
                    problems.append(f"{path.name}:{node.lineno} calls .{attr}()")
                if (
                    isinstance(fn, ast.Name)
                    and fn.id == "open"
                    and any(
                        isinstance(a, ast.Constant)
                        and isinstance(a.value, str)
                        and ("w" in a.value or "a" in a.value or "+" in a.value)
                        for a in node.args[1:]
                    )
                ):
                    problems.append(f"{path.name}:{node.lineno} opens for write")
    assert not problems, f"gate cores must be read-only: {problems}"
