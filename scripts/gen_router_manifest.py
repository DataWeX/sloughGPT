#!/usr/bin/env python3
"""Generate ``apps/api/server/routers/_manifest.py`` — the router mount table.

    python scripts/gen_router_manifest.py           # rewrite the manifest
    python scripts/gen_router_manifest.py --check   # CI: fail if it is stale

WHY THIS EXISTS (card 57b0ea27, deliverable 2)
----------------------------------------------
``_router_names`` in ``routers/__init__.py`` was a hand-maintained central
list — the switchyard the *Endpoint & Transport Rule* retires. Membership and
mount order still have to be written down somewhere, but they should be
**derived from the filesystem and checked**, not typed by hand and trusted.

The route *projections* themselves are not here: ``create_router`` emits a
route fragment from its descriptor, and each router module carries its own.
This file is only the table that says which modules get imported, in what
order. Adding a capability means dropping a file in ``routers/`` and
re-running this generator — never editing a shared list.

ORDER IS LOAD-BEARING — NEVER SORT IT
-------------------------------------
FastAPI's ``include_router`` is first-match-wins at runtime while OpenAPI is
last-wins, so mount order decides which router owns a colliding path. The
``/chat`` split needs ``inference`` mounted before ``chat``
(see ``tests/server/test_inference_router.py``); alphabetical order would put
``chat`` first and silently fork the spec from actual behaviour.

So reconciliation **preserves existing order**, appends newly discovered
modules (sorted, so growth is deterministic) and drops removed ones. It never
re-sorts.

EXCLUSIONS
----------
``PRE_LIFESPAN``  mounted directly in ``main.py`` so they answer *during*
                  model load; listing them here would register them twice.
                  Self-verified below against ``main.py`` — if one stops
                  being registered there, the check fails loudly.
no module-level ``router``
                  e.g. ``routers/api_keys.py``: a library module
                  (``ApiKeyManager``) whose HTTP class is never mounted in
                  production. The loader's ``getattr(mod, "router")`` would
                  raise on it anyway.

USAGE AS A GATE
---------------
``--check`` writes nothing: it renders to memory and diffs against the file
on disk, exiting 1 when they differ. ``tests/server/test_router_manifest.py``
calls it so the staleness check runs in the normal test gate as well as CI.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROUTERS_DIR = REPO / "apps" / "api" / "server" / "routers"
MAIN_PY = REPO / "apps" / "api" / "server" / "main.py"
MANIFEST = ROUTERS_DIR / "_manifest.py"
INIT_PY = ROUTERS_DIR / "__init__.py"

# Mounted directly in main.py pre-lifespan (health/status answer while the
# model is still loading). Must NOT appear in the mount table.
PRE_LIFESPAN = frozenset({"health", "status", "consciousness", "dashboard"})

GENERATOR = "scripts/gen_router_manifest.py"


def _has_module_router(path: Path) -> bool:
    """True when the module binds a top-level ``router`` the loader can getattr."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError):
        return False
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == "router" for t in node.targets):
                return True
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "router":
                return True
    return False


def discover() -> set[str]:
    """Router modules that can be mounted, i.e. every file exposing ``router``."""
    if not ROUTERS_DIR.exists():
        raise SystemExit(f"setup error: missing {ROUTERS_DIR}")
    out: set[str] = set()
    for p in ROUTERS_DIR.glob("*.py"):
        if p.name.startswith("_"):  # __init__.py, _manifest.py — not routers
            continue
        if _has_module_router(p):
            out.add(p.stem)
    return out


def check_pre_lifespan() -> None:
    """Each PRE_LIFESPAN router must still be registered directly in main.py.

    Otherwise it is registered nowhere — excluded from the table *and* from
    the app, which would drop its routes with nothing to catch it.
    """
    try:
        src = MAIN_PY.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:  # pragma: no cover - setup error
        raise SystemExit(f"setup error: cannot read {MAIN_PY}: {exc}") from exc
    missing = sorted(name for name in PRE_LIFESPAN if not re.search(rf"\b{name}\b", src))
    if missing:
        raise SystemExit(
            "gen_router_manifest: PRE_LIFESPAN routers no longer referenced in "
            f"main.py: {missing} — they would be served nowhere. Fix PRE_LIFESPAN "
            "or restore the main.py registration."
        )


def existing_order() -> list[str]:
    """Current mount order: manifest first, else the legacy inline list.

    The ``__init__`` fallback only matters for a checkout predating the
    manifest; it is kept so the generator can bootstrap from either state.
    """
    if MANIFEST.exists():
        text = MANIFEST.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"ROUTER_MODULES[^=]*=\s*\((.*?)\)", text, re.DOTALL)
        if m:
            return re.findall(r"[\"']([^\"']+)[\"']", m.group(1))
    if INIT_PY.exists():
        text = INIT_PY.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"_router_names\s*=\s*\[(.*?)\]", text, re.DOTALL)
        if m:
            return re.findall(r"[\"']([^\"']+)[\"']", m.group(1))
    return []


def reconcile() -> list[str]:
    """Known order first, then newly discovered modules — never a re-sort."""
    check_pre_lifespan()
    candidates = discover() - PRE_LIFESPAN
    order = [n for n in existing_order() if n in candidates]
    seen = set(order)
    order.extend(sorted(candidates - seen))
    return order


def render(names: list[str]) -> str:
    body = "\n".join(f'    "{n}",' for n in names)
    return f'''"""API router mount table — GENERATED FILE, DO NOT EDIT.

Regenerate with ``python {GENERATOR}`` after adding, removing or reordering a
router. ``python {GENERATOR} --check`` fails when this file is stale and runs
as part of the test gate (``tests/server/test_router_manifest.py``).

Read by ``routers.get_all_routers()``, which still imports each module
**lazily** and isolates each with try/except — this file is a table of names,
not imports, so it costs nothing at cold start (it binds no router modules).

Order here is mount order and is load-bearing: ``inference`` must precede
``chat``. See the docstring of ``{GENERATOR}``.
"""

from __future__ import annotations

__all__ = ["ROUTER_MODULES"]

ROUTER_MODULES: tuple[str, ...] = (
{body}
)
'''


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="exit 1 if the manifest is stale")
    args = ap.parse_args(argv)

    names = reconcile()
    expected = render(names)

    if args.check:
        if not MANIFEST.exists():
            print(
                f"STALE: {MANIFEST.relative_to(REPO)} does not exist — run {GENERATOR}",
                file=sys.stderr,
            )
            return 1
        actual = MANIFEST.read_text(encoding="utf-8", errors="replace")
        if actual != expected:
            got = re.findall(r"[\"']([^\"']+)[\"']", actual)
            print(
                "STALE: routers/_manifest.py does not match the routers/ directory.\n"
                f"  in manifest only : {sorted(set(got) - set(names))}\n"
                f"  on disk only     : {sorted(set(names) - set(got))}\n"
                f"  order differs    : {got != names}\n"
                f"  fix: python {GENERATOR}",
                file=sys.stderr,
            )
            return 1
        print(f"ok: routers/_manifest.py is current ({len(names)} routers)")
        return 0

    MANIFEST.write_text(expected, encoding="utf-8")
    print(f"wrote {MANIFEST.relative_to(REPO)}: {len(names)} routers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
