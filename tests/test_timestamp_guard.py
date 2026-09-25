"""Repo invariant: production code must never append ``"Z"`` to ``isoformat()``.

``datetime.isoformat()`` already ends with a numeric offset (``+00:00``), so
``isoformat() + "Z"`` produces ``2026-09-24T09:47:33.835728+00:00Z`` — a string
JavaScript parses as ``Invalid Date``. That single expression caused the souls
page bug this repo fixed in card 054.

Every timestamp must instead come from :func:`domain.shared.utc_now_iso` (or
``to_iso``), which always emits ``...Z``.

The scan is AST-based so docstrings, comments, and string literals mentioning
the anti-pattern do not trip it.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = ("domain", "apps", "packages/core-py", "scripts")
SKIP_PARTS = {"__pycache__", ".venv", "venv", "node_modules", ".wt-merge"}


class _IsoPlusZ(ast.NodeVisitor):
    """Collect line numbers of ``<something>.isoformat() + "Z"`` expressions."""

    def __init__(self) -> None:
        self.hits: list[int] = []

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.Add):
            left = node.left
            right = node.right
            if (
                isinstance(left, ast.Call)
                and isinstance(left.func, ast.Attribute)
                and left.func.attr == "isoformat"
                and isinstance(right, ast.Constant)
                and right.value == "Z"
            ):
                self.hits.append(node.lineno)
        self.generic_visit(node)

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        """Catch the f-string spelling: ``f"{ts.isoformat()}Z"``."""
        values = node.values
        for i, part in enumerate(values):
            if not isinstance(part, ast.FormattedValue):
                continue
            expr = part.value
            if (
                isinstance(expr, ast.Call)
                and isinstance(expr.func, ast.Attribute)
                and expr.func.attr == "isoformat"
            ):
                nxt = values[i + 1] if i + 1 < len(values) else None
                if (
                    isinstance(nxt, ast.Constant)
                    and isinstance(nxt.value, str)
                    and "Z" in nxt.value
                ):
                    self.hits.append(node.lineno)
        self.generic_visit(node)


def _production_py_files() -> list[Path]:
    files: list[Path] = []
    for name in SCAN_DIRS:
        base = ROOT / name
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            if SKIP_PARTS.intersection(path.parts):
                continue
            files.append(path)
    return sorted(files)


def test_no_bare_isoformat_plus_z_writers():
    offenders: list[str] = []
    for path in _production_py_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        # Cheap prefilter: only files that mention isoformat can be offenders.
        if "isoformat" not in text:
            continue
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError:
            continue
        finder = _IsoPlusZ()
        finder.visit(tree)
        for line in finder.hits:
            offenders.append(f"{path.relative_to(ROOT)}:{line}")

    assert not offenders, (
        "datetime.isoformat() + 'Z' writes '+00:00Z', which JS renders as "
        "'Invalid Date'. Use domain.shared.utc_now_iso() / to_iso() instead. "
        f"Offenders: {offenders}"
    )
