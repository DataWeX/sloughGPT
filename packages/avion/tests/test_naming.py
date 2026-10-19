"""Naming guard — ``avion`` is canonical; ``arken``/``voyager`` are shims.

The rename already landed, so the risk now is a *new* import creeping back
under one of the retired names. These tests fail the moment that happens.

Scan strategy: a line-level regex prefilters 2000+ files in ~1s, and only
the handful of candidates are parsed with the AST — full-AST over the whole
tree would cost 16s per test, and a regex alone would count a ``from arken
import (`` line split across two lines as prose.
"""

from __future__ import annotations

import ast
import re
import subprocess
import tomllib
from pathlib import Path

PACKAGES = Path(__file__).resolve().parents[2]
REPO_ROOT = PACKAGES.parent
AVION_DIR = PACKAGES / "avion"
#: The voyager shim is the only place the retired names may appear in an
#: import — it re-exports the whole chain (voyager → arken → avion).
VOYAGER_SHIM = PACKAGES / "voyager"

RETIRED = ("arken", "voyager")
_IMPORT_LINE = re.compile(r"^\s*(?:from\s+([A-Za-z_][\w.]*)\s+import|import\s+([A-Za-z_][\w.]*))")

_scan_cache: dict[str, list[tuple[Path, int]]] | None = None


def _tracked_py_files() -> list[Path]:
    """Every tracked .py in the repo (falls back to packages/ if no git)."""
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "ls-files", "*.py"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        out = None
    if out is not None and out.returncode == 0 and out.stdout.strip():
        return [REPO_ROOT / line for line in out.stdout.splitlines() if line]
    return list(AVION_DIR.rglob("*.py"))


def _line_imports_module(line: str) -> str | None:
    """Module name if this line opens an import of a retired package."""
    match = _IMPORT_LINE.match(line)
    if not match:
        return None
    return match.group(1) or match.group(2)


def _retired_imports(files: list[Path]) -> dict[str, list[tuple[Path, int]]]:
    """``{module: [(file, lineno), ...]}`` for real import statements only."""
    hits: dict[str, list[tuple[Path, int]]] = {name: [] for name in RETIRED}
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        # cheap prefilter: does any line even look like the import we want?
        candidate = False
        for line in text.splitlines():
            module = _line_imports_module(line)
            if module and any(module == name or module.startswith(f"{name}.") for name in RETIRED):
                candidate = True
                break
        if not candidate:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            for module in modules:
                for name in RETIRED:
                    if module == name or module.startswith(f"{name}."):
                        hits[name].append((path, node.lineno))
    return hits


def scan() -> dict[str, list[tuple[Path, int]]]:
    """Repo-wide retired-name imports, computed once per test session."""
    global _scan_cache
    if _scan_cache is None:
        _scan_cache = _retired_imports(_tracked_py_files())
    return _scan_cache


def test_import_avion_resolves():
    import avion

    assert avion.__name__ == "avion"


def test_pyproject_name_is_avion():
    pyproject = AVION_DIR / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    assert data["project"]["name"] == "avion"


def test_scan_finds_the_expected_shim_import():
    """The guard must not pass by finding nothing to check."""
    assert scan()["arken"], "expected the voyager shim to import arken"
    assert all(
        path == VOYAGER_SHIM / "src" / "voyager" / "__init__.py" for path, _ in scan()["arken"]
    )


def test_no_arken_imports_outside_the_voyager_shim():
    strays = [(path, line) for path, line in scan()["arken"] if VOYAGER_SHIM not in path.parents]
    assert strays == [], f"import arken outside the shim: {strays}"


def test_no_voyager_imports_outside_the_voyager_shim():
    strays = [(path, line) for path, line in scan()["voyager"] if VOYAGER_SHIM not in path.parents]
    assert strays == [], f"import voyager outside the shim: {strays}"


def test_arken_shim_reexports_from_canonical_avion():
    """The shim must point at avion, not drift into its own copy."""
    src = (PACKAGES / "arken" / "src" / "arken" / "__init__.py").read_text(encoding="utf-8")
    modules = {
        node.module
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "avion" in modules


def test_voyager_shim_still_targets_arken_chain():
    """voyager → arken → avion; both shims must keep re-exporting."""
    src = (VOYAGER_SHIM / "src" / "voyager" / "__init__.py").read_text(encoding="utf-8")
    modules = {
        node.module
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "arken" in modules
