"""Contract: frontend http-client call paths must resolve to served routes.

Scans apps/web apiGet/apiPost/apiPut/apiDelete/apiPatch literals (single call
per path is enough — paths, not call sites, are the contract) and checks each
against the probe-app route enumeration shared with test_routers_doc.

Drift found 2026-09-29 (10 paths, 4 clusters — see KNOWN_MISSING): the
baseline SHRINKS only (scoped to this branch: vm-console/checkpoint-compare calls exist only on the concurrent session's branch, not here): new missing paths fail; paths that start resolving
must be removed from KNOWN_MISSING in the same change that fixes them.
/api/* is served by Vite's apiRoutesPlugin middleware (dev), not uvicorn —
excluded by prefix, not by baseline.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_METHODS = ("GET", "POST", "PUT", "DELETE", "PATCH")
_ROOT = Path(__file__).resolve().parents[2]
_WEB = _ROOT / "apps" / "web"

# 2026-09-29 baseline: frontend calls with no matching backend route.
KNOWN_MISSING = frozenset(
    {
        # collections feature: backend API fully removed, UI still calls it
        ("GET", "/collections"),
        ("POST", "/collections/create"),
        ("POST", "/collections/run"),
        ("DELETE", "/collections/{param}"),
        # knowledge detail page + categorize shape mismatch
        # (backend: POST /knowledge/categorize without {id})
        ("GET", "/knowledge/{param}"),
        ("POST", "/knowledge/{param}/categorize"),
    }
)

_CALL = re.compile(
    r"\b(apiGet|apiPost|apiPut|apiDelete|apiPatch)"
    r"(?:\s*<[^<>]*(?:<[^<>]*>|[^<>])*>)?\s*\(\s*"
    r"(`(?:\\.|[^`\\])*`|'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\")"
)
_QUERYVAR = re.compile(
    r"^\s*(qs|query|params?|q|search|filter|sort|cursor|page|limit|offset|"
    r"suffix|usp|searchParams|rest)\s*$",
    re.I,
)


def _norm(p: str) -> str:
    p = re.sub(r"\{[^}]+\}", "{param}", p.split("?")[0])
    return p.rstrip("/") or "/"


def _variants(lit: str) -> set[str]:
    s = lit[1:-1]
    s = re.sub(r"\$\{[^}]*$", "", s)  # dangling nested-template tail
    out = { _norm(re.sub(r"\$\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", "{param}", s)) }

    def rm(m: re.Match[str]) -> str:
        var = re.match(r"[A-Za-z_$][A-Za-z0-9_$]*", m.group(0)[2:])
        if var and _QUERYVAR.match(var.group(0).lstrip("$")):
            return ""
        return "{param}"

    out.add(_norm(re.sub(r"\$\{(?:[^{}]|\$\{[^{}]*\})*\}", rm, s)))
    return out


def _web_calls() -> list[tuple[str, str]]:
    calls: list[tuple[str, str]] = []
    for f in list(_WEB.glob("**/*.ts")) + list(_WEB.glob("**/*.tsx")):
        if "node_modules" in f.parts or ".test." in f.name or "dist" in f.parts:
            continue
        try:
            txt = f.read_text()
        except OSError:
            continue
        for m in _CALL.finditer(txt):
            calls.append((m.group(1).replace("api", "").upper(), m.group(2)))
    return calls


@pytest.fixture(scope="module")
def missing_paths() -> set[tuple[str, str]]:
    import main as mainmod
    from fastapi import FastAPI
    from routers import get_all_routers
    from training.router import router as training_router

    probe = FastAPI()
    for r in get_all_routers():
        probe.include_router(r)
    probe.include_router(training_router)
    for name in (
        "_health_router",
        "_consciousness_router",
        "_status_router",
        "_dashboard_router",
    ):
        probe.include_router(getattr(mainmod, name))
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        paths = probe.openapi().get("paths", {})
    live = {
        (_m.upper(), _norm(p))
        for p, ops in paths.items()
        if not p.startswith("/api/")
        for _m in ops
        if _m.upper() in _METHODS
    }
    live_by_path: dict[str, set[str]] = {}
    for m, p in live:
        live_by_path.setdefault(p, set()).add(m)

    missing: set[tuple[str, str]] = set()
    matched = 0
    vite = 0
    for meth, lit in _web_calls():
        cands = _variants(lit)
        if any(c.startswith("/api/") for c in cands):
            vite += 1
            continue
        if any(meth in live_by_path.get(c, ()) for c in cands):
            matched += 1
        else:
            missing.add((meth, next(iter(cands))))
    assert matched >= 450, f"only {matched} matched — web/live scan is broken"
    assert vite >= 5, f"only {vite} vite-middleware calls found — scan broken"
    return missing


def test_no_new_unresolvable_frontend_calls(
    missing_paths: set[tuple[str, str]],
) -> None:
    new = sorted(missing_paths - KNOWN_MISSING)
    assert not new, (
        f"{len(new)} frontend call path(s) have no served route — fix the "
        f"call or add the backend route: {new}"
    )


def test_known_missing_baseline_shrinks(missing_paths: set[tuple[str, str]]) -> None:
    fixed = sorted(KNOWN_MISSING - missing_paths)
    assert not fixed, (
        f"{len(fixed)} baseline entr(y/ies) now resolve — remove them from "
        f"KNOWN_MISSING: {fixed}"
    )
