"""Repo invariant: production timestamps must be JS-parseable *and* offset-carrying.

Three historical failure modes are scanned for (all AST-based, so docstrings,
comments, and string literals mentioning them do not trip the guard):

1. ``<ts>.isoformat() + "Z"`` — ``isoformat()`` already ends with a numeric
   offset (``+00:00``), so appending ``Z`` produces ``…+00:00Z``, a string
   JavaScript parses as ``Invalid Date``. That single expression caused the
   souls page bug this repo fixed in card 054.
2. ``datetime.now().isoformat()`` — naive local time with no offset, read as
   the *viewer's* zone by JS (every non-matching zone sees a shifted clock).
3. ``datetime.fromtimestamp(t).isoformat()`` without ``tz=`` — same naive
   local time, different spelling.

Every timestamp must come from :func:`domain.shared.utc_now_iso` (or ``to_iso``),
which always emits ``...Z``.
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


class _NaiveNowIso(ast.NodeVisitor):
    """Collect line numbers of ``datetime.now().isoformat()`` (naive, no tz).

    A naive timestamp carries no offset, so JavaScript reads it as the
    *viewer's* local time — every viewer outside the server's zone (WAT, UTC+1
    here) sees a shifted clock. Write ``utc_now_iso()`` instead. ``now()`` with
    a timezone argument is fine and not reported.
    """

    def __init__(self) -> None:
        self.hits: list[int] = []

    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "isoformat"
            and isinstance(node.func.value, ast.Call)
            and isinstance(node.func.value.func, ast.Attribute)
            and node.func.value.func.attr == "now"
            and not node.func.value.args
            and not node.func.value.keywords
        ):
            self.hits.append(node.lineno)
        self.generic_visit(node)


class _NaiveFromtimestampIso(ast.NodeVisitor):
    """Collect ``datetime.fromtimestamp(t).isoformat()`` without a ``tz=``.

    ``fromtimestamp`` with no timezone yields naive *local* time, so the value
    carries no offset for JavaScript to anchor on — the same shifted-clock bug
    as ``datetime.now().isoformat()``. Pass ``tz=UTC`` (see ``training/feeds.py``)
    or serialize with ``to_iso`` instead.
    """

    def __init__(self) -> None:
        self.hits: list[int] = []

    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "isoformat"
            and isinstance(node.func.value, ast.Call)
            and isinstance(node.func.value.func, ast.Attribute)
            and node.func.value.func.attr == "fromtimestamp"
            and all(kw.arg != "tz" for kw in node.func.value.keywords)
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


def _collect(make_visitor) -> list[str]:
    """Run *make_visitor* over every scanned production file; return hits.

    Only files mentioning ``isoformat`` can match, so the text prefilter keeps
    the AST pass cheap.
    """
    offenders: list[str] = []
    for path in _production_py_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if "isoformat" not in text:
            continue
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError:
            continue
        finder = make_visitor()
        finder.visit(tree)
        offenders.extend(f"{path.relative_to(ROOT)}:{line}" for line in finder.hits)
    return offenders


def test_no_bare_isoformat_plus_z_writers():
    offenders = _collect(_IsoPlusZ)

    assert not offenders, (
        "datetime.isoformat() + 'Z' writes '+00:00Z', which JS renders as "
        "'Invalid Date'. Use domain.shared.utc_now_iso() / to_iso() instead. "
        f"Offenders: {offenders}"
    )


def test_no_naive_now_isoformat_writers():
    offenders = _collect(_NaiveNowIso)

    assert not offenders, (
        "datetime.now().isoformat() emits a naive local timestamp with no "
        "offset; JS parses it as the viewer's local time and every non-matching "
        "zone sees a shifted clock. Use domain.shared.utc_now_iso() instead. "
        f"Offenders: {offenders}"
    )


def test_no_naive_fromtimestamp_isoformat_writers():
    offenders = _collect(_NaiveFromtimestampIso)

    assert not offenders, (
        "datetime.fromtimestamp(t).isoformat() without tz= serializes naive "
        "local time; JS reads it as the viewer's zone. Pass tz=UTC or use "
        "domain.shared.to_iso() instead. "
        f"Offenders: {offenders}"
    )
