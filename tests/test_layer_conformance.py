"""Layer conformance — routers and controllers never reach past the domain facade.

Card b22e878a (GROUP A + GROUP C): the "grep -c _internal <router> == 0" rule
lived only in the playbook as prose, which is exactly how it drifted from
"only shell/vm" to 13 routers. This module is the executable form of that
rule: an AST import scan (so prose/docstring mentions of ``_internal`` do not
count) over every router and controller, plus a positive control that the
controller layer really does delegate through ``domain.<feature>`` facades.

Card 20261101_140447 (routers out of ``domain.*_internal*``): the directory
list was itself the remaining hole — playbook rule 1 lets a feature package
(``apps/api/server/<feature>/``) grow sub-routers outside ``routers/``, and
the ``training/`` package's 8 sub-routers imported ``_internal`` while every
test here stayed green. The route surface is now scanned *structurally* for
everything outside the two directories: a production module is route surface
if it constructs an ``APIRouter``, applies a route decorator, or calls
``add_api_route`` — whichever way it declares.

Target seam: router -> controller -> engine -> domain.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SERVER_ROOT = REPO / "apps" / "api" / "server"
ROUTER_DIR = REPO / "apps" / "api" / "server" / "routers"
CONTROLLER_DIR = REPO / "apps" / "api" / "server" / "controllers"

# Excluded zones per card b22e878a: world_render was another session's
# concurrent-refactor territory (it historically DID use _internal) — it has
# scanned clean since that refactor landed, so the zone is empty as of
# 20261101_140447 (2026-11-09). api_keys was dropped when merged into
# routers/settings.py (route-map card 5/7, 20261101_154056); shell.py + vm.py
# were dropped when vm.py was merged into shell.py (route-map card 6/7,
# 20261101_154057) — neither has any AST ``_internal`` import (shell's lone
# mention is a docstring). The mechanism stays for future concurrent zones;
# it may only ever shrink to this empty state, never grow back silently.
ROUTER_ALLOWLIST: frozenset[str] = frozenset()

# Route-declaration verbs: a decorator like ``@router.get`` is the other
# declaration idiom (same shape check_contract --static uses for the
# hand-built-route ratchet).
_ROUTE_VERBS = frozenset({"get", "post", "put", "delete", "patch", "head", "options", "websocket"})


def _module_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.glob("*.py") if p.name != "__init__.py")


def _is_route_surface(tree: ast.AST) -> bool:
    """True when *tree* declares HTTP routes (any of the three idioms)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "APIRouter":
                return True
            if isinstance(node.func, ast.Attribute) and node.func.attr in (
                "APIRouter",
                "add_api_route",
            ):
                return True
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Attribute) and decorator.attr in _ROUTE_VERBS:
                    return True
    return False


def _route_surface_files_outside_router_dirs() -> list[Path]:
    """Structural route surface outside ``routers/`` and ``controllers/``.

    Why structural: playbook rule 1 gives every feature package its own
    ``router.py`` (``apps/api/server/training/`` has 13 route-declaring
    modules), and the directory list never saw them — this card exists
    because the 8 ``_internal``-importing training sub-routers passed every
    test here. ``tests/`` is excluded: the facade-binding rule deliberately
    lets test modules patch ``domain.*._internal`` names.
    """
    router_dir_files = set(_module_files(ROUTER_DIR)) | set(_module_files(CONTROLLER_DIR))
    surface: list[Path] = []
    for path in sorted(SERVER_ROOT.rglob("*.py")):
        if "tests" in path.parts or "__pycache__" in path.parts:
            continue
        if path in router_dir_files or path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if _is_route_surface(tree):
            surface.append(path)
    return surface


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


def test_route_surfaces_outside_router_dirs_do_not_import_internal() -> None:
    """Structural sweep: every route-declaring module outside ``routers/`` and
    ``controllers/`` is facade-clean too (the 8 training sub-routers of card
    20261101_140447 live here — the directory list never scanned them)."""
    hits: list[str] = []
    for path in _route_surface_files_outside_router_dirs():
        rel = path.relative_to(REPO).as_posix()
        for entry in _internal_imports(path):
            hits.append(f"{rel} :: {entry.split(':', 1)[1]}")
    assert not hits, (
        "route surface must import domain facades, not domain.*._internal:\n" + "\n".join(hits)
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
