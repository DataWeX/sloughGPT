"""Layer conformance — routers and controllers never reach past the domain facade.

Card b22e878a (GROUP A + GROUP C): the "grep -c _internal <router> == 0" rule
lived only in the playbook as prose, which is exactly how it drifted from
"only shell/vm" to 13 routers. This module is the executable form of that
rule: an AST import scan (so prose/docstring mentions of ``_internal`` do not
count) over every router and controller, plus a positive control that the
controller layer really does delegate through ``domain.<feature>`` facades.

Target seam: router -> controller -> engine -> domain.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROUTER_DIR = REPO / "apps" / "api" / "server" / "routers"
CONTROLLER_DIR = REPO / "apps" / "api" / "server" / "controllers"

# Excluded zones per card b22e878a: shell/vm/world_render are another
# session's concurrent-refactor territory (they historically DO use
# _internal), and api_keys is deliberately unmounted.
ROUTER_ALLOWLIST = frozenset({"shell.py", "vm.py", "world_render.py", "api_keys.py"})


def _module_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.glob("*.py") if p.name != "__init__.py")


def _internal_imports(path: Path) -> list[str]:
    """Return every import line in *path* that reaches into a ``_internal`` namespace."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and "_internal" in node.module:
                hits.append(f"{path.name}:{node.lineno}: from {node.module} import ...")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if "_internal" in alias.name:
                    hits.append(f"{path.name}:{node.lineno}: import {alias.name}")
    return hits


def _facade_import_count(directory: Path) -> int:
    """Count imports that land on a ``domain.<feature>`` package facade."""
    count = 0
    for path in _module_files(directory):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("domain.")
                and "_internal" not in node.module
            ):
                count += 1
    return count


def test_routers_do_not_import_internal() -> None:
    """No router outside the excluded zone imports ``_internal`` (acceptance 1)."""
    hits: list[str] = []
    for path in _module_files(ROUTER_DIR):
        if path.name in ROUTER_ALLOWLIST:
            continue
        hits.extend(_internal_imports(path))
    assert not hits, "routers must import domain facades, not domain.*._internal:\n" + "\n".join(
        hits
    )


def test_controllers_do_not_import_internal() -> None:
    """No controller imports ``_internal`` — controllers proxy through facades."""
    hits: list[str] = []
    for path in _module_files(CONTROLLER_DIR):
        hits.extend(_internal_imports(path))
    assert not hits, (
        "controllers must import domain facades, not domain.*._internal:\n" + "\n".join(hits)
    )


def test_controllers_import_domain_facade() -> None:
    """Positive control: the controller layer really does delegate to facades.

    The negative tests above would also pass if controllers simply stopped
    touching the domain layer entirely; this asserts the swap seams exist.
    """
    count = _facade_import_count(CONTROLLER_DIR)
    assert count >= 10, (
        f"expected controllers to reach the domain layer via >=10 facade imports, found {count}"
    )
