"""Shared-workspace governance contract tests (AGENTS.md cards 080/081).

Two invariants, enforced mechanically so they cannot rot into prose:

1. **Index freshness** — every runnable app in ``apps/`` must be
   registered in ``apps/README.md``, every library in ``packages/`` in
   ``packages/README.md``, and every doc linked from ``docs/INDEX.md``
   must exist. New app + forgot the index = red test.
2. **One shared toolchain** — ``node_modules``/``.venv``/lockfiles may
   only exist at the allowlisted shared roots. A second install tree
   (the 7.5G ``apps/mobile`` duplicate was the cautionary tale) fails.

Scope note: ``.wt-*`` directories are independent git worktree
checkouts owned by other sessions; they are outside this scan.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Snapshot allowlist — current legitimate installs, each with a reason.
# Adding an entry requires a reason in the same change (and ideally a
# card); anything NOT listed here is a governance violation.
ALLOWED_NODE_MODULES = {
    ROOT / "node_modules",  # npm workspace root install
    ROOT / "apps" / "web" / "node_modules",  # Vite app (own lockfile tree)
    ROOT / "apps" / "mobile" / "node_modules",  # RN app — regenerable, do NOT rebuild
    ROOT / "packages" / "strui" / "node_modules",  # published package, own lock
    ROOT / "packages" / "eslint-config" / "node_modules",  # published config, own deps
    ROOT
    / "packages"
    / "sdk-ts"
    / "typescript-sdk"
    / "node_modules",  # npm SDK, CI job test-sdk-ts runs npm ci here
    ROOT / ".opencode" / "node_modules",  # opencode plugin dependencies
}
ALLOWED_LOCKS = {
    ROOT / "package-lock.json",
    ROOT / "apps" / "web" / "package-lock.json",
    ROOT / "apps" / "mobile" / "package-lock.json",
    ROOT / "packages" / "strui" / "package-lock.json",
    ROOT / "packages" / "sdk-ts" / "typescript-sdk" / "package-lock.json",  # npm SDK own lock
    ROOT / ".opencode" / "package-lock.json",
}
ALLOWED_VENVS = {
    ROOT / ".venv",  # THE project venv — every session/worktree uses this one
}

_SKIP_DIRS = {"__pycache__", "node_modules", ".git"}


def _children(base: Path, patterns: tuple[str, ...]) -> list[Path]:
    out: list[Path] = []
    for pat in patterns:
        out.extend(p for p in base.glob(pat) if not any(part in _SKIP_DIRS for part in p.parts))
    return sorted(set(out))


def _dir_names(base: Path) -> list[str]:
    return sorted(
        d.name for d in base.iterdir() if d.is_dir() and not d.name.startswith((".", "_"))
    )


# ── 1. Index freshness ──────────────────────────────────────────────────────


def test_every_app_is_indexed() -> None:
    """A new dir under apps/ must be registered in apps/README.md."""
    readme = (ROOT / "apps" / "README.md").read_text(encoding="utf-8")
    missing = [n for n in _dir_names(ROOT / "apps") if f"`{n}/`" not in readme]
    assert not missing, (
        f"apps not indexed in apps/README.md: {missing} — add an entry in the "
        "same change that adds the app (AGENTS.md: keep the shared indexes current)"
    )


def test_every_package_is_indexed() -> None:
    """A new dir under packages/ must be registered in packages/README.md."""
    readme = (ROOT / "packages" / "README.md").read_text(encoding="utf-8")
    missing = [n for n in _dir_names(ROOT / "packages") if f"`{n}/`" not in readme]
    assert not missing, (
        f"packages not indexed in packages/README.md: {missing} — add a row to "
        "the package table in the same change that adds the package"
    )


def test_docs_index_links_resolve() -> None:
    """Every .md referenced in docs/INDEX.md must exist (docs/, docs/<subdir>/, or root).

    Subpaths are allowed (e.g. ``design/DESIGN_SYSTEM.md``); a bare filename must
    resolve directly under ``docs/`` or the repo root.
    """
    index = (ROOT / "docs" / "INDEX.md").read_text(encoding="utf-8")
    refs = set(re.findall(r"([A-Za-z0-9_][A-Za-z0-9_./-]*\.md)", index))
    missing = sorted(
        r for r in refs if not (ROOT / "docs" / r).exists() and not (ROOT / r).exists()
    )
    assert not missing, f"docs/INDEX.md references missing docs: {missing}"


# ── 2. One shared toolchain ─────────────────────────────────────────────────


def _flag_strays(name: str, allowed: set[Path]) -> list[str]:
    hits: list[Path] = []
    for base in (ROOT, ROOT / "apps", ROOT / "packages"):
        # depth 1 + 2 covers root, apps/<x>, packages/<x> artifacts
        for pat in (name, f"*/{name}", f"*/*/{name}"):
            for p in base.glob(pat):
                rel = p.relative_to(ROOT).parts[:-1]  # parents only
                if any(part in _SKIP_DIRS or part.startswith(".wt-") for part in rel):
                    continue  # inside an install/.git, or a .wt-* worktree checkout
                if p not in allowed:
                    hits.append(p)
    return sorted(str(p.relative_to(ROOT)) for p in set(hits))


def test_no_stray_node_modules() -> None:
    strays = _flag_strays("node_modules", ALLOWED_NODE_MODULES)
    assert not strays, (
        f"node_modules outside the shared roots: {strays} — reuse the root "
        "install (AGENTS.md: packages are shared, never rebuilt)"
    )


def test_no_stray_lockfiles() -> None:
    strays = _flag_strays("package-lock.json", ALLOWED_LOCKS)
    assert not strays, (
        f"package-lock.json outside the shared roots: {strays} — a second lock "
        "means a second install tree; reuse the shared one"
    )


def test_no_stray_venvs() -> None:
    strays = _flag_strays(".venv", ALLOWED_VENVS)
    assert not strays, (
        f".venv outside the shared root: {strays} — use THE project venv "
        "(AGENTS.md: always use the project venv)"
    )
