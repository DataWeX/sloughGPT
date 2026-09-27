"""Router → domain facades must resolve; `_internal` reach-ins must shrink.

Build Order #4 ("point routers at engines, not internals"). Phase A fixed
the 7 facade imports that raised ImportError at runtime; this test keeps
them fixed and ratchets the remaining `_internal` reach-ins per file so
phases B/C can only move the number down.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROUTERS_DIR = Path(__file__).resolve().parents[1] / "routers"

# Per-file `_internal` reach-in counts at phase A (2026-09-26, total 131).
# Update DOWNWARD only as phases B/C route routers through facades/engines.
_INTERNAL_BASELINE = {
    "agents.py": 6,
    "benchmark.py": 2,
    "consciousness.py": 21,
    "dashboard.py": 3,
    "files.py": 1,
    "lora_eval.py": 3,
    "memory.py": 6,
    "models.py": 5,
    "registry.py": 2,
    "self_train.py": 1,
    "settings.py": 17,
    "shell.py": 3,
    "status.py": 1,
    "system.py": 2,
    "tenants.py": 2,
    "tokens.py": 1,
    "tools.py": 1,
    "users.py": 2,
    "vm.py": 4,
    "workspaces.py": 5,
    "world_render.py": 6,
}


def _router_files() -> list[Path]:
    return sorted(ROUTERS_DIR.glob("*.py"))


def _domain_imports() -> list[tuple[str, int, str, str]]:
    found: list[tuple[str, int, str, str]] = []
    for path in _router_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.level == 0
                and node.module
                and node.module.startswith("domain.")
            ):
                for alias in node.names:
                    found.append((path.name, node.lineno, node.module, alias.name))
    return found


def _internal_reachins() -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in _router_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        n = 0
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.level == 0
                and node.module
                and node.module.startswith("domain.")
                and "._internal" in node.module
            ):
                n += 1
        if n:
            counts[path.name] = n
    return counts


# Known-dead imports: code paths that reference repositories which have never
# existed anywhere (workspace-scoped dataset/knowledge cleanup+search in
# workspaces.py). They are inert behind try/except and need a product decision
# (add workspace scoping or drop the blocks) — see dev-note card 062.
_KNOWN_DEAD = {
    (
        "workspaces.py",
        "domain.dataset._internal.repository",
        "DatasetRepository",
    ),
    (
        "workspaces.py",
        "domain.learner._internal.knowledge",
        "KnowledgeRepository",
    ),
}


def test_every_domain_facade_import_resolves():
    failures = []
    for fname, line, module, name in _domain_imports():
        if name == "*" or (fname, module, name) in _KNOWN_DEAD:
            continue
        try:
            # Real `from X import Y` semantics: attribute lookup with
            # submodule-import fallback (getattr-only would false-flag).
            exec(f"from {module} import {name}", {})
        except Exception as exc:  # noqa: BLE001 - report every broken facade
            failures.append(f"{fname}:{line}  from {module} import {name}  ->  {exc}")
    assert not failures, (
        "Routers import domain symbols that do not exist on the public facade "
        "(they would raise ImportError at runtime):\n" + "\n".join(failures)
    )


def test_internal_reachins_do_not_grow():
    current = _internal_reachins()
    regressions = []
    for fname, baseline in _INTERNAL_BASELINE.items():
        now = current.get(fname, 0)
        if now > baseline:
            regressions.append(f"{fname}: {now} > baseline {baseline}")
    new_files = sorted(set(current) - set(_INTERNAL_BASELINE))
    for fname in new_files:
        regressions.append(f"{fname}: new `_internal` reach-ins ({current[fname]})")
    assert not regressions, (
        "New `domain.*._internal` reach-ins in routers (route through the "
        "public facade or an engine instead):\n" + "\n".join(regressions)
    )
