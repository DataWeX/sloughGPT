"""Ratchet: executor config must not reference deleted or moved paths.

A path that appears in an agent definition, a skill, a Makefile target or a
shell script is a *promise to the runner* -- "run this", "edit that", "open
this doc". When the path is later moved or deleted, the instruction silently
keeps working as text while failing as work: the test stops running, the doc
stops opening, the script stops launching. That is how a fix stops being
enforced and the code walks back to a state we already repaired.

Scope is deliberately *executor config*, not every file in the repo:

    .opencode/agents/*.md      what sessions are told to run
    .opencode/skills/*/SKILL.md
    scripts/**                 what the shell actually executes
    pytest.ini                 what gets collected

Historical citations in ``docs/`` are out of scope -- ``docs/REPO_STRUCTURE_
MIGRATION.md`` exists precisely to name paths that no longer exist.

Three context rules keep this honest instead of noisy (a guard that cries wolf
is a guard that gets ignored):

1. **cwd-relative.** ``make test-py ARGS="tests/test_x.py"`` runs under
   ``cd packages/core-py``, so ``tests/test_x.py`` means
   ``packages/core-py/tests/test_x.py``. Both readings are accepted; a path
   fails only when *neither* resolves.
2. **Probes and provenance are not claims.** ``if p.is_dir()`` is asking, not
   promising; "Ported from ``scripts/old.py``" is history.
3. **Destinations may not exist yet.** ``Serve from apps/web/public/buildroot/``
   names a directory to create -- allowed while its parent exists, so a moved
   or deleted *tree* still fails.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Repo-relative roots whose literals we can resolve against a checkout.
ROOTS = (r"\.opencode", "packages", "domain", "tests", "apps", "scripts", "docs")
PATH_RE = re.compile(
    r"(?<![\w/.-])(?:" + "|".join(ROOTS) + r")/[A-Za-z0-9_./<>*-]+"
)

# A bare version tail is a domain identifier, not a path: ``apiVersion: apps/v1``
# is a Kubernetes manifest whose apiGroup happens to collide with our ``apps/``.
VERSION_TAIL_RE = re.compile(r"^v\d+$")

# Illustrative filenames used as doctest/doc placeholders, not real targets.
EXAMPLE_NAMES = frozenset({"test_file.py", "test_foo.py", "file.py", "x.py"})

# The line is *asking* whether a path exists rather than asserting it does.
PROBE_MARKERS = (
    "is_dir", "exists", "[ -d", "test -d", "os.path.exists",
    "Path(", "glob(", "listdir", "lexists",
)

# The line records where something came from, or names a destination to build.
PROVENANCE_MARKERS = (
    "ported from", "relocated", "no longer", "migrated", "census",
    "orphan", "historical", "provenance", "was removed",
)
CREATION_MARKERS = (
    "mkdir", "cp ", " of=", "tee ", "touch ", "serve from",
    "copy ", "place ", "extract to", "output/",
)


@lru_cache(maxsize=1)
def _scan_files() -> tuple[Path, ...]:
    files: list[Path] = [
        *sorted((REPO / ".opencode" / "agents").glob("*.md")),
        *sorted((REPO / ".opencode" / "skills").glob("*/SKILL.md")),
        REPO / "pytest.ini",
    ]
    for p in sorted((REPO / "scripts").rglob("*")):
        if p.is_file() and p.suffix in {".sh", ".py", ".bash"}:
            files.append(p)
    return tuple(p for p in files if p.is_file())


@lru_cache(maxsize=1)
def _make_cd_dirs() -> dict[str, str]:
    """Map Makefile target -> directory its recipe cds into before running."""
    out: dict[str, str] = {}
    makefile = REPO / "Makefile"
    if not makefile.is_file():
        return out
    current: str | None = None
    for line in makefile.read_text(errors="replace").splitlines():
        if line.startswith("\t"):
            if current:
                m = re.search(r"\bcd\s+([A-Za-z0-9_./-]+)\s+&&", line)
                if m and current not in out:
                    out[current] = m.group(1)
            continue
        m = re.match(r"^([A-Za-z0-9_./-]+)\s*:(?!=)", line)
        if m:
            current = m.group(1)
    return out


def _bases_for(line: str) -> tuple[Path, ...]:
    """Directories a path literal on this line may be relative to."""
    m = re.search(r"\bcd\s+([A-Za-z0-9_./-]+)\s+&&", line)
    if m:
        return (REPO, (REPO / m.group(1)).resolve())
    m = re.search(r"\bmake\s+(test-py[\w-]*|test-repo-root)", line)
    if m:
        cd = _make_cd_dirs().get(m.group(1))
        if cd:
            return (REPO, (REPO / cd).resolve())
    return (REPO,)


def _normalize(raw: str) -> str | None:
    """Reduce a matched literal to the concrete path it asserts."""
    text = raw.rstrip(".,;:)]}'\">")
    if not text:
        return None
    concrete: list[str] = []
    for i, part in enumerate(text.split("/")):
        if any(ch in part for ch in "<>*?"):
            break
        concrete.append(part)
        # `tests/helpers.RangeHandler` is module.attr, not a file on disk.
        if i == len(text.split("/")) - 1 and "." in part:
            stem, _, ext = part.rpartition(".")
            if not (1 <= len(ext) <= 8 and ext.isalnum() and ext.islower() and stem):
                concrete[-1] = stem
                break
    if len(concrete) < 2:
        return None
    return "/".join(concrete)


def _resolves(rel: str, bases: tuple[Path, ...], trailing_dir: bool) -> bool:
    for base in bases:
        target = base / rel
        # ``pkg/tests/helpers`` means ``helpers.py`` -- a module reference, and
        # ``.../helpers.RangeHandler`` truncates to the module before its attr.
        if target.exists() or Path(f"{target}.py").exists():
            return True
        if trailing_dir and target.parent.exists():
            return True  # a destination to be created, not a moved tree
    return False


def _stale_references() -> dict[str, list[str]]:
    stale: dict[str, list[str]] = {}
    for path in _scan_files():
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError:
            continue
        for lineno, line in enumerate(lines, 1):
            low = line.lower()
            if any(m in low for m in PROBE_MARKERS):
                continue
            if any(m in low for m in PROVENANCE_MARKERS):
                continue
            if any(m in low for m in CREATION_MARKERS):
                continue
            bases = _bases_for(line)
            for raw in PATH_RE.findall(line):
                rel = _normalize(raw)
                if not rel:
                    continue
                if Path(rel).name in EXAMPLE_NAMES:
                    continue
                if VERSION_TAIL_RE.match(Path(rel).name):
                    continue  # ``apiVersion: apps/v1`` -- Kubernetes, not a tree
                if _resolves(rel, bases, trailing_dir=raw.rstrip().endswith("/")):
                    continue
                stale.setdefault(rel, []).append(
                    f"{path.relative_to(REPO)}:{lineno}"
                )
    return stale


def test_executor_config_references_paths_that_exist() -> None:
    stale = _stale_references()
    if not stale:
        return
    report = "\n".join(
        f"  {rel}\n    at {', '.join(sorted(set(where))[:4])}"
        for rel, where in sorted(stale.items())
    )
    raise AssertionError(
        f"{len(stale)} path reference(s) point at files that are gone. Either "
        f"the path moved (repoint the reference) or the target was deleted "
        f"(remove the instruction):\n{report}"
    )


def test_scanned_executor_config_is_non_empty() -> None:
    """Guard the guard: an empty scan would pass while checking nothing."""
    files = _scan_files()
    assert len(files) >= 30, f"only {len(files)} executor-config files scanned"
    rels = {str(p.relative_to(REPO)) for p in files}
    assert ".opencode/agents/systems-engineer.md" in rels
    assert "pytest.ini" in rels
