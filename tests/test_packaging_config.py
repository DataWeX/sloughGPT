"""Packaging config consistency — card 56cf354c.

The dual-tree consolidation (cards 20260924_047..058) deleted
``packages/core-py/domains`` and moved ``domain`` to the repo root, but the
setuptools config stayed behind: ``package-dir`` still mapped the whole tree
under ``packages/core-py``, so discovery found ``domain`` (via a "." root) and
the map pointed it at the deleted ``packages/core-py/domain`` —
``python setup.py egg_info`` hard-failed and the documented
``pip install -e ".[dev]"`` could not run.

These tests pin the invariants whose violation caused that:

1. pyproject ``package-dir`` root and ``packages.find`` roots are ONE root —
   a package discovered outside ``package_dir[""]`` maps to a path that does
   not exist.
2. ``setup.py`` declares no packaging config of its own (pyproject is the
   single source of truth; the old duplicated copy had drifted).
3. Discovery actually finds the shipped top-levels, including the
   ``[project.scripts]`` entry point package, and no test packages.
4. The literal trap from the card: ``python setup.py egg_info`` exits 0 and
   the generated ``top_level.txt`` contains ``domain``.
"""

import ast
import subprocess
import sys
import tomllib
from pathlib import Path

from setuptools import find_packages

REPO = Path(__file__).resolve().parents[1]


def _setuptools_cfg() -> dict:
    with open(REPO / "pyproject.toml", "rb") as fh:
        return tomllib.load(fh)["tool"]["setuptools"]


def test_package_dir_and_find_share_one_source_root() -> None:
    cfg = _setuptools_cfg()
    root = cfg["package-dir"][""]
    where = cfg["packages"]["find"]["where"]
    assert where == [root], (
        f"packages.find roots {where} must equal package-dir[''] {root!r}: "
        "a package discovered under a different root maps to a directory "
        "that does not exist (card 56cf354c egg_info hard-fail)"
    )
    assert (REPO / root / "domain" / "__init__.py").exists(), (
        f"{root}/domain is not a package on disk — discovery would ship nothing"
    )


def test_setup_py_declares_no_packaging_config() -> None:
    """setup.py must stay a thin shim over pyproject.toml, not a 2nd copy."""
    tree = ast.parse((REPO / "setup.py").read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "setup"
    ]
    assert len(calls) == 1, "expected exactly one setup() call"
    assert not calls[0].args, "setup() must not pass positional packaging args"
    assert not calls[0].keywords, (
        "setup() must not pass packages/package_dir — pyproject.toml is the "
        "single source of truth (drift here broke egg_info, card 56cf354c)"
    )


def test_discovery_finds_shipped_toplevels() -> None:
    find_cfg = _setuptools_cfg()["packages"]["find"]
    pkgs = find_packages(
        where=find_cfg["where"][0],
        include=tuple(find_cfg["include"]),
        exclude=tuple(find_cfg["exclude"]),
    )
    assert "domain" in pkgs, "domain* is the shipped core tree"
    assert "apps.cli" in pkgs
    assert "apps.cli.src" in pkgs, "[project.scripts] sloughgpt = apps.cli.src.cli:main"
    assert not [p for p in pkgs if p.endswith(".tests") or ".tests." in p], (
        "test packages must not ship"
    )
    assert not [p for p in pkgs if p.startswith("apps.web")]


def test_setup_py_egg_info_succeeds(tmp_path: Path) -> None:
    """The literal trap from card 56cf354c (non-mutating: writes only to tmp)."""
    proc = subprocess.run(
        [sys.executable, "setup.py", "egg_info", "--egg-base", str(tmp_path)],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    egg_info_dirs = list(tmp_path.glob("*.egg-info"))
    assert len(egg_info_dirs) == 1, f"expected one egg-info dir, got {egg_info_dirs}"
    tops = (egg_info_dirs[0] / "top_level.txt").read_text().split()
    assert "domain" in tops, f"top_level.txt missing domain: {tops}"
