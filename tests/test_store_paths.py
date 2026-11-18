"""Guard: hand-counted repo paths must resolve INSIDE the repository root.

The defect class (found 2026-11-15, card 20261115_112756): ``os.path.join(
os.path.dirname(__file__), "..", ...)`` and ``Path(__file__).resolve().parents[N]``
encode the module's directory depth by hand.  When a refactor moves the file
one level (``domains.X`` -> ``domain.X._internal``) the count silently points
*above* the repo root, and the application starts reading/writing a store
outside the checkout (the /agents page showed 0 agents while 81 real agent
files sat untouched in ``data/agents``), or onto a nonexistent directory
(checkpoint listing pointed at ``apps/api/models/...`` which does not exist).

Three gates:
1. every ``dirname(__file__) + ".."*`` climb in production code lands in-repo;
2. every ``Path(__file__).resolve().parents[N]`` lands in-repo;
3. explicit regression anchors: the seven measured sites resolve to exactly
   their intended in-repo targets.  Anchors resolve the *source expression*
   (never importing the module — no import side effects), supporting both the
   legacy hand-counted forms and the ``find_repo_root`` marker-walk form, so
   a future move of the declaring module re-runs the count instead of
   trusting it.

Deliberate escapes are allowlisted per (file, levels) with a reason — the
only one is db_pool's pre-fix legacy-data pointer (card 055).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# os.path.join(os.path.dirname(__file__) | Path(__file__).parent, ".."/seg, ...)
_JOIN_CLIMB_RE = re.compile(
    r"""os\.path\.join\(\s*(?:os\.path\.dirname\(__file__\)|Path\(__file__\)\.parent)"""
    r"""((?:\s*,\s*"\.\.")+)"""
)

# Full join argument list (all literal segments, tails included).
_JOIN_FULL_RE = re.compile(
    r"""os\.path\.join\(\s*(?:os\.path\.dirname\(__file__\)|Path\(__file__\)\.parent)"""
    r"""((?:\s*,\s*"[^"]*"\s*)+)\)"""
)

# Path(__file__).resolve().parents[N]
_PARENTS_CLIMB_RE = re.compile(r"""Path\(__file__\)\.resolve\(\)\.parents\[(\d+)\]""")
_PARENTS_FULL_RE = re.compile(
    r"""Path\(__file__\)\.resolve\(\)\.parents\[(\d+)\]((?:\s*/\s*"[^"]*"\s*)*)"""
)

# find_repo_root(...) marker-walk form — the fix pattern:
#   os.path.join(find_repo_root(__file__), "data", "agents")   (str tails)
#   find_repo_root(Path(__file__)) / "models" / "x"            (Path tails)
#   str(find_repo_root(__file__))                              (no tails)
# One nesting level allowed inside find_repo_root(...) so both
# find_repo_root(__file__) and find_repo_root(Path(__file__).resolve()) match
# in full — a plain [^)]* would stop at Path(__file__'s ')' and silently drop
# the "/ seg" tails, resolving to the bare repo root instead of the target.
_FIND_CALL = r"find_repo_root\((?:[^()]|\([^()]*\))*\)"
_FIND_JOIN_RE = re.compile(rf"""os\.path\.join\(\s*{_FIND_CALL}\s*((?:,\s*"[^"]*"\s*)+)\)""")
_FIND_PATH_RE = re.compile(rf"""{_FIND_CALL}\s*((?:/\s*"[^"]*"\s*)*)""")

# Directories never worth scanning for path constants.
_SKIP_PARTS = {
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    ".git",
    "build",
    "dist",
}

# Deliberate escapes: (relative file path, climb levels) -> reason.
# db_pool (card 055 / commit 6c5c0453b): parents[5] intentionally points
# above the repo so copy-once adoption can find pre-fix data and copy it in.
# The live store is already resolved via find_repo_root; only this legacy
# pointer is allowed to leave the checkout.
_ALLOWLISTED_ESCAPES: dict[tuple[str, int], str] = {
    (
        "apps/api/server/infrastructure/db_pool.py",
        5,
    ): "deliberate pre-fix legacy-data pointer (card 055 / commit 6c5c0453b)",
}

# Top-level dirs of production code worth scanning.
_SCAN_TOPS = ("domain", "apps", "packages", "scripts")


def _production_py_files() -> list[Path]:
    files: list[Path] = []
    for top in _SCAN_TOPS:
        base = REPO_ROOT / top
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_PARTS]
            for name in filenames:
                if name.endswith(".py"):
                    files.append(Path(dirpath) / name)
    return sorted(files)


def _climb(start: Path, levels: int) -> Path:
    current = start
    for _ in range(levels):
        current = current.parent
    return current


def _escapes_repo(target: Path) -> bool:
    """True when *target* is neither the repo root nor below it."""
    resolved = target.resolve()
    return resolved != REPO_ROOT and REPO_ROOT not in resolved.parents


def _allowlisted(path: Path, levels: int) -> bool:
    rel = str(path.relative_to(REPO_ROOT))
    return (rel, levels) in _ALLOWLISTED_ESCAPES


def _resolve_join_args(start: Path, args: list[str]) -> Path:
    current = start
    for arg in args:
        current = current.parent if arg == ".." else current / arg
    return current


def _resolve_parents(file: Path, level: int, tails: list[str]) -> Path:
    # Path(p).parents[0] == p.parent, so parents[N] == N+1 climbs from p itself.
    current = _climb(file, level + 1)
    for tail in tails:
        current = current / tail
    return current


def _marker_root(start: Path) -> Path:
    """Mirror of find_repo_root's primary rule (apps/ + packages/ markers)."""
    for parent in start.parents:
        if (parent / "apps").is_dir() and (parent / "packages").is_dir():
            return parent
    raise AssertionError(f"no repo root (apps+packages) found above {start}")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_join_climb_paths_stay_inside_repo() -> None:
    """Every ``dirname(__file__) + ".."*`` climb must land inside the repo."""
    violations: list[str] = []
    checked = 0
    for path in _production_py_files():
        try:
            source = _read(path)
        except OSError:  # pragma: no cover - unreadable file
            continue
        for match in _JOIN_CLIMB_RE.finditer(source):
            levels = match.group(1).count('".."')
            checked += 1
            if _allowlisted(path, levels):
                continue
            climbed = _climb(path.parent, levels)
            if _escapes_repo(climbed):
                line = source.count("\n", 0, match.start()) + 1
                violations.append(
                    f"{path.relative_to(REPO_ROOT)}:{line} climbs {levels}x '..' "
                    f"-> {climbed} (outside repo root {REPO_ROOT})"
                )
    assert checked > 0, "scan found no join-climb paths — pattern regex is broken"
    assert not violations, "path constants escape the repo root:\n" + "\n".join(violations)


def test_file_parents_paths_stay_inside_repo() -> None:
    """Every ``Path(__file__).resolve().parents[N]`` must land inside the repo."""
    violations: list[str] = []
    checked = 0
    for path in _production_py_files():
        try:
            source = _read(path)
        except OSError:  # pragma: no cover - unreadable file
            continue
        for match in _PARENTS_CLIMB_RE.finditer(source):
            level = int(match.group(1))
            checked += 1
            if _allowlisted(path, level):
                continue
            climbed = _climb(path, level + 1)  # parents[N] == N+1 climbs from file
            if _escapes_repo(climbed):
                line = source.count("\n", 0, match.start()) + 1
                violations.append(
                    f"{path.relative_to(REPO_ROOT)}:{line} parents[{level}] "
                    f"-> {climbed} (outside repo root {REPO_ROOT})"
                )
    assert checked > 0, "scan found no parents[N] paths — pattern regex is broken"
    assert not violations, "path constants escape the repo root:\n" + "\n".join(violations)


# --- Regression anchors for the seven measured sites (card 20261115_112756) --
# (declaring module, intended target relative to repo root).
_ANCHORS: list[tuple[str, Path]] = [
    ("domain/agents/_internal/system.py", Path("data/agents")),
    ("domain/agents/_internal/run_history.py", Path("data/agent_runs")),
    ("domain/agents/_internal/routines.py", Path("data/routines")),
    ("domain/auth/_internal/repositories.py", Path("")),
    ("apps/api/server/training/router.py", Path("models") / "auto-training"),
    # db.py has two store paths: the mogdb src dir (sys.path) and the
    # default DB dir (was a Path.cwd() climb — the 7th site, same class).
    ("apps/cli/src/groups/db.py", Path("packages/mogdb/src")),
    ("apps/cli/src/groups/db.py", Path("data") / "mogdb"),
    # db_pool: the LIVE root must be marker-resolved in-repo (the "/ data"
    # join happens downstream via _get_repo_root(), which this source scan
    # does not follow); only the legacy parents[5] pointer above may escape.
    ("apps/api/server/infrastructure/db_pool.py", Path("")),
]


def test_store_targets_resolve_to_intended_in_repo_paths() -> None:
    """The measured store targets must resolve to their intended in-repo paths.

    Resolution is derived from the source expression on disk (never importing
    the declaring module), so this stays a pure-function check with no import
    side effects, and a future refactor of the expression must keep resolving
    to the same target.  Both legacy hand-counted forms and the
    ``find_repo_root`` marker-walk form are resolved.
    """
    problems: list[str] = []
    for rel, expected_rel in _ANCHORS:
        declaring = REPO_ROOT / rel
        if not declaring.is_file():
            problems.append(f"{rel}: declaring module not found (moved?)")
            continue
        source = _read(declaring)
        expected = (REPO_ROOT / expected_rel).resolve() if expected_rel.parts else REPO_ROOT
        resolved: set[Path] = set()

        # Legacy: os.path.join(os.path.dirname(__file__), ...)
        for match in _JOIN_FULL_RE.finditer(source):
            args = re.findall(r'"([^"]*)"', match.group(1))
            resolved.add(_resolve_join_args(declaring.parent, args).resolve())
        # Legacy: Path(__file__).resolve().parents[N] / ...
        for match in _PARENTS_FULL_RE.finditer(source):
            level = int(match.group(1))
            tails = re.findall(r'"([^"]*)"', match.group(2))
            resolved.add(_resolve_parents(declaring, level, tails).resolve())
        # Fix pattern: find_repo_root(...) with str-style join tails.
        for match in _FIND_JOIN_RE.finditer(source):
            tails = re.findall(r'"([^"]*)"', match.group(1))
            resolved.add(_marker_root(declaring).joinpath(*tails).resolve())
        # Fix pattern: find_repo_root(...) with Path-style "/ seg" tails (or bare).
        for match in _FIND_PATH_RE.finditer(source):
            tails = re.findall(r'"([^"]*)"', match.group(1))
            resolved.add(_marker_root(declaring).joinpath(*tails).resolve())

        if not resolved:
            problems.append(f"{rel}: no recognized path expression found — anchor needs updating")
        elif expected not in resolved:
            got = ", ".join(sorted(str(p) for p in resolved))
            problems.append(f"{rel}: intended {expected} not among resolved paths [{got}]")
    assert not problems, "store-path anchors broken:\n" + "\n".join(problems)
