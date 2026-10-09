"""Static guard: store-of-record writes must go through an atomic primitive.

Raw truncate-in-place writes (``open(path, "w")`` / ``Path.write_text`` on
the live file) can leave a store empty or torn if the process dies
mid-write: the old content is already gone while the new content may never
reach disk. Stores must write **tmp → fsync → ``os.replace`` → dir fsync**
via ``mogdb.durability`` (journals/mirrors) or ``app_planner.atomic``
(planner files).

Structural counterpart to the runtime crash battery in
``packages/mogdb/tests/test_durability.py``: the battery proves the
primitive survives SIGKILL; this test stops new raw writes from bypassing
it. Exemptions are (file, qualname)-scoped with a reason, and must match a
real raw-write site — stale exemptions fail too (card faa1cfa7).
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCAN_ROOTS = (
    REPO / "packages" / "mogdb" / "src",
    REPO / "packages" / "app-planner" / "src",
)

# (relpath, enclosing qualname or None = whole file) -> reason.
EXEMPT: dict[tuple[str, str | None], str] = {
    ("packages/mogdb/src/mogdb/durability.py", None): (
        "the primitive itself: O_APPEND single-syscall appends, quarantine "
        "sidecar appends, tmp -> fsync -> replace + dir fsync"
    ),
    ("packages/app-planner/src/app_planner/atomic.py", None): (
        "the planner primitive: mkstemp -> fsync -> replace + dir fsync"
    ),
    ("packages/app-planner/src/app_planner/core.py", "NoteStore.export_all"): (
        "user-specified export output (regenerable projection)"
    ),
    ("packages/app-planner/src/app_planner/kanban.py", "cli_main"): (
        "export command output (regenerable projection)"
    ),
}


def _mode_of(node: ast.Call) -> str | None:
    """Extract the open()/fdopen() mode argument, if it is a literal."""
    if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
        v = node.args[1].value
        if isinstance(v, str):
            return v
    for kw in node.keywords:
        if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
            v = kw.value.value
            if isinstance(v, str):
                return v
    return None


class _WriteScanner(ast.NodeVisitor):
    """Collect every write-mode file call with its enclosing qualname."""

    def __init__(self, rel: str) -> None:
        self.rel = rel
        self.stack: list[str] = []
        self.sites: list[tuple[str, int, str]] = []  # (qualname, lineno, what)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        what: str | None = None
        if isinstance(func, ast.Name) and func.id == "open":
            mode = _mode_of(node)
            if mode and mode[0] in ("w", "a", "x"):
                what = f"open(mode={mode!r})"
        elif isinstance(func, ast.Attribute) and func.attr in (
            "write_text",
            "write_bytes",
            "fdopen",
        ):
            if func.attr == "fdopen":
                mode = _mode_of(node)
                if not (mode and mode[0] in ("w", "a", "x")):
                    what = None
                else:
                    what = f"os.fdopen(mode={mode!r})"
            else:
                what = f".{func.attr}()"
        if what:
            self.sites.append((".".join(self.stack) or "<module>", node.lineno, what))
        self.generic_visit(node)


def _scan() -> dict[str, list[tuple[str, int, str]]]:
    """All raw-write sites per file (relative posix path)."""
    found: dict[str, list[tuple[str, int, str]]] = {}
    for root in SCAN_ROOTS:
        for path in sorted(root.rglob("*.py")):
            rel = path.relative_to(REPO).as_posix()
            visitor = _WriteScanner(rel)
            visitor.visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
            if visitor.sites:
                found[rel] = visitor.sites
    return found


def test_scan_roots_exist() -> None:
    """The guard is worthless if its roots silently stop matching."""
    for root in SCAN_ROOTS:
        assert root.is_dir(), f"scan root missing: {root}"


def test_exemptions_match_real_sites() -> None:
    """No stale exemptions: every (file, qualname) must still raw-write."""
    found = _scan()
    for (rel, qual), _reason in EXEMPT.items():
        assert (REPO / rel).is_file(), f"exemption file gone: {rel}"
        if qual is None:
            assert rel in found, f"whole-file exemption no longer needed: {rel}"
            continue
        sites = found.get(rel, [])
        assert any(q == qual for q, _, _ in sites), (
            f"stale exemption {rel}:{qual} — no raw-write site remains there; remove it"
        )


def test_store_writes_go_through_atomic_primitive() -> None:
    """No raw write-mode calls outside the primitives/export sites."""
    found = _scan()
    violations: list[str] = []
    for rel, sites in found.items():
        for qual, lineno, what in sites:
            if (rel, None) in EXEMPT or (rel, qual) in EXEMPT:
                continue
            violations.append(f"{rel}:{lineno} [{qual}] {what}")
    assert not violations, (
        "raw truncate-in-place store writes found — use "
        "mogdb.durability.atomic_write / app_planner.atomic.atomic_write_text "
        "(or add an EXEMPT entry with a reason if it is a regenerable export):\n"
        + "\n".join(violations)
    )
