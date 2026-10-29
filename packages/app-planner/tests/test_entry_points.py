"""Console-script declarations must resolve to importable callables (card 830190a4).

After the CLI consolidation moved the real CLI to ``app_planner.cli``, both
planner packages still declared ``planner = planner.cli:main`` — a module that
no longer exists anywhere — so every ``planner``/``kanban`` console script and
``python -m planner`` raised ImportError. This test walks every
``console_scripts`` declaration in the repo (pyproject ``[project.scripts]``
and setup.py ``entry_points``) and resolves each target: the module imports
and the attribute is callable. No declaration may dangle.
"""

from __future__ import annotations

import ast
import importlib
import sys
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


def _declared_entries() -> list[tuple[str, str, Path]]:
    """(command name, module:attr target, declaring file) across the repo."""
    entries: list[tuple[str, str, Path]] = []

    pyprojects = [REPO / "pyproject.toml", *sorted((REPO / "packages").glob("*/pyproject.toml"))]
    for pyproject in pyprojects:
        if not pyproject.is_file():
            continue
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        for name, target in (data.get("project", {}).get("scripts") or {}).items():
            entries.append((name, target, pyproject))

    for setup_py in sorted((REPO / "packages").glob("*/setup.py")):
        tree = ast.parse(setup_py.read_text(encoding="utf-8"))
        for call in ast.walk(tree):
            if not isinstance(call, ast.Call):
                continue
            for kw in call.keywords:
                if kw.arg != "entry_points" or not isinstance(kw.value, ast.Dict):
                    continue
                for key, value in zip(kw.value.keys, kw.value.values):
                    if not (isinstance(key, ast.Constant) and key.value == "console_scripts"):
                        continue
                    if isinstance(value, (ast.List, ast.Tuple)):
                        for elt in value.elts:
                            if isinstance(elt, ast.Constant) and "=" in elt.value:
                                name, _, target = elt.value.partition("=")
                                entries.append((name.strip(), target.strip(), setup_py))
    return entries


def _bootstrap_sys_path(entries: list[tuple[str, str, Path]]) -> None:
    """Make every declaring package's code importable for this process.

    Layouts differ (src/ vs flat), and the Makefile PYTHONPATH covers only
    packages/app-planner/src — so each declaring directory contributes its
    src/ (if any) and its own dir (flat layout) exactly once.
    """
    roots: set[Path] = set()
    for _, _, declared_in in entries:
        pkg_dir = declared_in.parent
        roots.add(pkg_dir)
        src = pkg_dir / "src"
        if src.is_dir():
            roots.add(src)
    roots.add(REPO / "apps" / "cli" / "src")  # root pyproject sloughgpt entry
    roots.add(REPO)
    for root in sorted(str(r) for r in roots):
        if root not in sys.path:
            sys.path.insert(0, root)


ENTRIES = _declared_entries()
_bootstrap_sys_path(ENTRIES)


def test_repo_declares_console_scripts() -> None:
    assert len(ENTRIES) >= 8, f"expected the known repo-wide script set, got {ENTRIES}"


@pytest.mark.parametrize(
    ("name", "target", "declared_in"),
    ENTRIES,
    ids=[
        f"{name}@{declared_in.name}:{declared_in.parent.name}"
        for name, target, declared_in in ENTRIES
    ],
)
def test_console_script_target_resolves(name: str, target: str, declared_in: Path) -> None:
    module_name, _, attr = target.partition(":")
    module = importlib.import_module(module_name)
    assert callable(getattr(module, attr)), f"{declared_in}: {name} -> {target} is not callable"


def test_python_dash_m_planner_imports() -> None:
    """`python -m planner` must not ImportError (it imported the deleted planner.cli).

    Loaded from the repo's own source file by path: an editable install of
    ``planner`` may point at a different checkout (meta-path finders beat
    sys.path), and this gate must test what the repo declares, not what the
    machine happens to have installed.
    """
    import importlib.util

    path = REPO / "packages" / "planner" / "src" / "planner" / "__main__.py"
    spec = importlib.util.spec_from_file_location("_planner_main_under_test", path)
    assert spec is not None and spec.loader is not None, f"no import spec for {path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # top-level import must not raise
    assert callable(module.main)
