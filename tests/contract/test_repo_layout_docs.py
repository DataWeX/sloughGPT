"""Contract: repo-layout docs stay truthful against the live tree.

Guards README.md and docs/STRUCTURE.md — the drift class found in the
2026-09-29 seam-5 audit:

- backticked repo paths that no longer exist (scripts/legacy/ was already gone),
- markdown links to moved/renamed docs,
- stale commands (pytest `-n auto` needs pytest-xdist, not installed locally),
- stale counts (consciousness system pages),
- README API table rows with no served route,
- Makefile/`package.json` script claims with no target behind them,
- the `datasets/` tree entry (absent until first import/download).

Any change to the real infrastructure that invalidates these docs must update
them in the same change.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_DOCS = {
    "README": _REPO / "README.md",
    "STRUCTURE": _REPO / "docs" / "STRUCTURE.md",
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
    "infra/",
    "data/",
)
_SKIP_SUBSTRINGS = ("*", " ", "=", "\t")
# Lines that reference paths in a negative/historical sense are allowed.
_NEGATIVE_CONTEXT = (
    "removed",
    "deleted",
    "superseded",
    "no shims",
    "stub",
    "formerly",
    "moved to",
    "predate",
)

_STALE_CLAIMS = {
    "README": {
        "-n auto": "pytest-xdist is not installed locally — the flag errors",
        "26 pages": "consciousness page count drift (now 25)",
    },
    "STRUCTURE": {
        "scripts/legacy/": "directory no longer exists",
        "make help": "Makefile has no help target",
        "one-off legacy snippets": "scripts/legacy/ was removed",
    },
}

# Root files docs/STRUCTURE.md names in its repo-root paragraph.
_ROOT_CLAIMS = (
    "pyproject.toml",
    "README.md",
    "config.yaml",
    "cli.py",
    "sloughgpt_colab.ipynb",
    "package.json",
    "Makefile",
    "verify.sh",
    "install.sh",
    "run.sh",
)

# Top-level entries from the README project-structure tree (datasets handled
# separately: it is runtime-generated, not checked into the repo).
_TREE_CLAIMS = (
    "apps/api/server",
    "apps/web",
    "apps/cli",
    "apps/mobile",
    "apps/gateway",
    "apps/data",
    "domain",
    "packages/core-py",
    "packages/strui",
    "packages/mogdb",
    "packages/sdk-py",
    "packages/sdk-ts",
    "packages/standards",
    "tests",
    "scripts",
)


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
                if token.startswith("./"):
                    token = token[2:]
                if not token.startswith(_PATH_PREFIXES):
                    continue
                if not (_REPO / token).exists():
                    missing.append(f"{key}: {token}")
    assert not missing, f"repo-layout docs reference dead paths: {missing}"


def test_markdown_links_resolve() -> None:
    broken: list[str] = []
    for key, path in _DOCS.items():
        base = path.parent
        for target in re.findall(r"\]\(([^)#]+)\)", path.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            if not (base / target).exists():
                broken.append(f"{key}: {target}")
    assert not broken, f"repo-layout docs contain dead links: {broken}"


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
        "_repo_layout_routes", _REPO / "tests" / "contract" / "test_routers_doc.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._live_routes()


def test_readme_api_table_live() -> None:
    paths = {p for _, p in _live_routes()}
    rows = re.findall(r"\| `(GET|POST|PUT|DELETE|PATCH)` \| `(/[^`]+)` \|", _read("README"))
    assert rows, "no API rows found in README.md table"
    dead = [
        f"{meth} {path}"
        for meth, path in rows
        if not any(r == path or r.startswith(path.rstrip("/") + "/") for r in paths)
    ]
    assert not dead, f"README API table rows with no served route: {dead}"


def test_structure_root_and_readme_tree() -> None:
    problems: list[str] = []
    for name in _ROOT_CLAIMS:
        if not (_REPO / name).exists():
            problems.append(f"STRUCTURE root file missing: {name}")
    for name in _TREE_CLAIMS:
        if not (_REPO / name).is_dir():
            problems.append(f"README tree directory missing: {name}")
    if not (_REPO / "datasets").exists():
        # Runtime-generated: the tree entry must say so instead of implying a
        # checked-in directory.
        line = next(
            (ln for ln in _read("README").splitlines() if re.search(r"datasets/\s+#", ln)),
            "",
        )
        if "runtime" not in line:
            problems.append(
                "README tree lists datasets/ but the directory does not exist "
                "and the entry does not mark it runtime"
            )
    assert not problems, problems


def test_structure_claims_have_targets() -> None:
    problems: list[str] = []
    # npm scripts claimed in docs/STRUCTURE.md (`dev:stack`, `test:repo-root`)
    scripts = json.loads((_REPO / "package.json").read_text(encoding="utf-8")).get("scripts", {})
    for name in set(re.findall(r"`([a-z]+:[a-z-]+)`", _read("STRUCTURE"))):
        if name not in scripts:
            problems.append(f"package.json script missing: {name}")
    # Make targets claimed as `make <target>`
    makefile = (_REPO / "Makefile").read_text(encoding="utf-8")
    for name in set(re.findall(r"\bmake ([a-z0-9][a-z0-9-]*)", _read("STRUCTURE"))):
        if not re.search(rf"^{re.escape(name)}:", makefile, re.MULTILINE):
            problems.append(f"Makefile target missing: {name}")
    assert not problems, problems


def test_readme_consciousness_count_matches_reality() -> None:
    c_root = _REPO / "apps" / "web" / "app" / "(app)" / "consciousness"
    actual = len([d for d in c_root.iterdir() if d.is_dir() and (d / "page.tsx").exists()])
    m = re.search(r"consciousness system with (\d+) pages", _read("README"))
    assert m, "README.md consciousness '…with N pages' claim not found"
    doc_count = int(m.group(1))
    assert doc_count == actual, (
        f"README says consciousness system with {doc_count} pages but reality is "
        f"{actual} sub-pages (page.tsx) — update the count"
    )
