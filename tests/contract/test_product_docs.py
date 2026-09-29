"""Contract: product docs stay truthful against the live codebase.

Guards docs/PRODUCT_ENGINEERING.md, ROADMAP.md, FEATURES.md, INDEX.md —
the drift class found in the 2026-09-29 seam-4 audit:

- backticked repo paths that no longer exist (dead runners, deleted packages),
- stale headline counts (pages/routers/consciousness sub-pages),
- feature-table API prefixes with no served route and frontend pages with no
  directory,
- known-false claims that were fixed in the truth-up (58 routers, 32 pages,
  voyager 201 tests, Click-era CLI commands, self-reported coverage tables),
- routers reintroducing `_internal` imports.

Any change to the real infrastructure that invalidates these docs must update
them in the same change.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_DOCS = {
    "PRODUCT": _REPO / "docs" / "PRODUCT_ENGINEERING.md",
    "ROADMAP": _REPO / "docs" / "ROADMAP.md",
    "FEATURES": _REPO / "docs" / "FEATURES.md",
    "INDEX": _REPO / "docs" / "INDEX.md",
}
_PATH_PREFIXES = (
    "tests/",
    "scripts/",
    "apps/",
    "packages/",
    "domain/",
    "docs/",
    ".github/",
    "config/",
)
_SKIP_SUBSTRINGS = ("*", " ", "=", "\t")
# Lines that reference paths in a negative/historical sense are allowed.
_NEGATIVE_CONTEXT = (
    "removed",
    "deleted",
    "superseded",
    "never re-created",
    "no shims",
    "stub",
    "formerly",
    "moved to",
    "predate",
)

_STALE_CLAIMS = {
    "PRODUCT": {
        "58 routers": "router file count drift",
        "27 sub-pages": "consciousness sub-page count drift",
        "22 router files": "old router-file claim",
    },
    "ROADMAP": {
        "32 pages": "web page count drift (now 109)",
        "201 tests": "voyager suite moved to avion (225 tests)",
        "324 files": "vitest file count drift",
        "3048 tests": "vitest test count drift",
    },
    "FEATURES": {
        "26 pages": "consciousness page count drift (now 25)",
        "apps/cli/cli.py": "entrypoint no longer exists (use ./sloughgpt)",
        "./sloughgpt datasets ": "plural group replaced by `dataset`",
        "autotrain": "Click-era top-level command removed",
        "feedback-export": "Click-era top-level command removed",
        "--dataset shakespeare": "old train flags predate the train group",
        "## Tools Pages\n": "section replaced by removal note",
        "| 100% |": "self-reported coverage table",
    },
}


def _read(key: str) -> str:
    return _DOCS[key].read_text(encoding="utf-8")


def test_referenced_paths_exist() -> None:
    missing: list[str] = []
    for key, path in _DOCS.items():
        for line in path.read_text(encoding="utf-8").splitlines():
            low = line.lower()
            if any(ctx in low for ctx in _NEGATIVE_CONTEXT):
                continue
            for token in re.findall(r"`([^`\n]+)`", line):
                token = token.strip().rstrip(".,;:")
                if any(s in token for s in _SKIP_SUBSTRINGS):
                    continue
                if not token.startswith(_PATH_PREFIXES):
                    continue
                if not (_REPO / token).exists():
                    missing.append(f"{key}: {token}")
    assert not missing, f"product docs reference dead paths: {missing}"


def test_no_stale_claims() -> None:
    hits: list[str] = []
    for key, claims in _STALE_CLAIMS.items():
        text = _read(key)
        for claim, why in claims.items():
            if claim in text:
                hits.append(f"{key}: {claim!r} ({why})")
    assert not hits, f"stale claims reintroduced: {hits}"


def _live_routes() -> set[tuple[str, str]]:
    # Reuse the probe enumeration from the routers-doc contract (same process,
    # no on-disk artifact): loads the sibling module by path — importlib test
    # mode does not put tests/contract on sys.path.
    spec = importlib.util.spec_from_file_location(
        "_product_docs_routes", _REPO / "tests" / "contract" / "test_routers_doc.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._live_routes()


def test_feature_table_api_prefixes_and_pages() -> None:
    text = _read("PRODUCT")
    paths = {p for _, p in _live_routes()}
    app_root = _REPO / "apps" / "web" / "app" / "(app)"
    problems: list[str] = []
    for line in text.splitlines():
        if not re.match(r"^\| *\d+ +\|", line):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 6:
            continue  # Experimental table has no prefix/page columns
        api_cell, page_cell = cells[4], cells[5]
        for tok in re.findall(r"`(/[^`?]+)`", api_cell):
            if not any(r == tok or r.startswith(tok.rstrip("/") + "/") for r in paths):
                problems.append(f"API prefix {tok} has no live route (row: {cells[1]})")
        for tok in re.findall(r"`(/[^`?]+)`", page_cell):
            slug = tok.strip("/").split("/")[0]
            if not (app_root / slug).exists():
                problems.append(f"frontend page {tok} does not exist (row: {cells[1]})")
    assert not problems, problems


def test_headline_counts_match_reality() -> None:
    pages = len(list((_REPO / "apps" / "web" / "app" / "(app)").rglob("page.tsx")))
    routers = len(
        [
            p
            for p in (_REPO / "apps" / "api" / "server" / "routers").glob("*.py")
            if p.name != "__init__.py"
        ]
    )
    m = re.search(r"Current: (\d+) frontend pages, (\d+) routers", _read("PRODUCT"))
    assert m, "PRODUCT_ENGINEERING.md headline 'Current: N frontend pages, M routers' not found"
    doc_pages, doc_routers = int(m.group(1)), int(m.group(2))
    assert (doc_pages, doc_routers) == (pages, routers), (
        f"headline says {doc_pages} pages/{doc_routers} routers but reality is "
        f"{pages} pages/{routers} router files — update the headline"
    )


def test_consciousness_counts_match_reality() -> None:
    c_root = _REPO / "apps" / "web" / "app" / "(app)" / "consciousness"
    actual = len([d for d in c_root.iterdir() if d.is_dir() and (d / "page.tsx").exists()])
    claims: list[int] = []
    fm = re.search(r"consciousness system[^.]*?(\d+) pages", _read("FEATURES"))
    assert fm, "FEATURES.md '…consciousness system… N pages' claim not found"
    claims.append(int(fm.group(1)))
    prod = _read("PRODUCT")
    m2 = re.search(r"Consciousness: (\d+) sub-pages", prod)
    assert m2, "PRODUCT_ENGINEERING.md dead-code row consciousness count not found"
    claims.append(int(m2.group(1)))
    m3 = re.search(r"(\d+) sub-pages still need flow review", prod)
    assert m3, "PRODUCT_ENGINEERING.md experimental-table consciousness count not found"
    claims.append(int(m3.group(1)))
    assert claims, "no consciousness counts found to verify"
    stale = [c for c in claims if c != actual]
    assert not stale, (
        f"consciousness counts {stale} do not match reality ({actual} sub-pages "
        "with page.tsx) — update the docs"
    )


def test_router_facade_claim_matches_reality() -> None:
    internal = 0
    for p in (_REPO / "apps" / "api" / "server" / "routers").glob("*.py"):
        for line in p.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if re.match(r"^(from|import) ", s) and "_internal" in s:
                internal += 1
    assert internal == 0, (
        f"{internal} `_internal` import(s) reintroduced under apps/api/server/routers/ "
        "— routers must import public domain facades only"
    )
    m = re.search(r"0 `_internal` imports across all (\d+) `routers/", _read("PRODUCT"))
    assert m, (
        "PRODUCT_ENGINEERING.md facade claim ('0 _internal imports across all N routers') not found"
    )
    router_files = [
        p
        for p in (_REPO / "apps" / "api" / "server" / "routers").glob("*.py")
        if p.name != "__init__.py"
    ]
    assert int(m.group(1)) == len(router_files), (
        "facade claim's router-file count drifts from apps/api/server/routers/*.py"
    )
