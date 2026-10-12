#!/usr/bin/env python3
"""Purge orphaned and stale __pycache__ bytecode.

Two failure modes, both inert (CPython keys off the source file, never the
cache) but both costly — every traceback, coverage report and pytest
assertion-rewritten frame reads ``co_filename`` out of the pyc, so a bad cache
cites paths that no longer exist:

* **orphaned** — no ``.py`` sibling remains. Left behind when a test file is
  deleted or moved out of the tree.
* **stale** — the ``.py`` sibling exists, but the ``co_filename`` embedded at
  compile time points somewhere that no longer exists. ``mv``/repo restructures
  preserve mtime+size, so CPython keeps using the pyc; its tracebacks cite the
  OLD path.

Ported from ``scripts/check_stale_pycache.py`` (81145e743, branch
feat/per-app-feature-flags) with the two classes separated, a dry-run-default
CLI, and an importable API safe enough to call from a pytest_sessionstart
guard.

Usage:
    python scripts/purge_stale_pyc.py                    # dry-run, tests/ + packages/
    python scripts/purge_stale_pyc.py tests/core-py      # dry-run, one root
    python scripts/purge_stale_pyc.py --apply            # delete
"""

from __future__ import annotations

import argparse
import marshal
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_ROOTS = (_REPO_ROOT / "tests", _REPO_ROOT / "packages")

# PEP 552 (Python 3.7+) writes a 16-byte header: magic, flags, timestamp, size.
# Legacy is 12: magic, timestamp, size. Trying both keeps this working across
# interpreters rather than pinning it to one.
_HEADER_OFFSETS = (16, 12)


def _co_filename(pyc: Path) -> str | None:
    """The source path a .pyc was compiled from, or None if unreadable."""
    try:
        data = pyc.read_bytes()
    except OSError:
        return None
    for offset in _HEADER_OFFSETS:
        try:
            code = marshal.loads(data[offset:])
        except Exception:
            continue
        name = getattr(code, "co_filename", None)
        if isinstance(name, str):
            return name
    return None


def _iter_pyc(root: Path) -> list[Path]:
    """Bytecode under ``root``, skipping foreign worktrees and vendored trees.

    Same reasoning as check_stale_pycache.py (81145e743): the repo root holds
    ``.wt-*`` worktrees, each a full checkout, and a filesystem walk counts
    every cache once per worktree (card 3dfbbb90). Untracked caches under
    ignored worktrees are somebody else's to clean.

    The worktree filter must test the path RELATIVE to root. Testing the
    absolute path silently skips everything when root itself lives inside a
    ``.wt-*`` worktree — every path under it contains that component — which
    reports "No orphaned bytecode" from a checkout that has plenty.
    """
    results = []
    for pyc in root.rglob("*.pyc"):
        try:
            rel_parts = pyc.relative_to(root).parts
        except ValueError:
            rel_parts = pyc.parts
        if any(part.startswith(".wt-") or part == "node_modules" for part in rel_parts):
            continue
        results.append(pyc)
    return results


def _source_sibling(pyc: Path) -> Path:
    """The ``.py`` this pyc would be keyed on: ``x/__pycache__/a.b.pyc`` -> ``x/a.py``."""
    base_dir = pyc.parent.parent if pyc.parent.name == "__pycache__" else pyc.parent
    stem = pyc.name.split(".")[0]
    return base_dir / f"{stem}.py"


@dataclass
class PurgeReport:
    """Classification of every pyc scanned. Lists hold (pyc, detail) pairs."""

    scanned: int = 0
    orphaned: list[tuple[Path, str]] = field(default_factory=list)
    stale: list[tuple[Path, str]] = field(default_factory=list)
    kept: int = 0
    deleted: int = 0
    skipped_tracked: int = 0

    @property
    def removable(self) -> list[tuple[Path, str]]:
        return self.orphaned + self.stale


def classify(roots: list[Path]) -> PurgeReport:
    """Sort bytecode under ``roots`` into orphaned / stale / kept. Never raises."""
    report = PurgeReport()
    for root in roots:
        if not root.is_dir():
            continue
        for pyc in _iter_pyc(root):
            report.scanned += 1
            sibling = _source_sibling(pyc)
            if not sibling.exists():
                # No source anywhere near the cache: orphaned, whatever the
                # header says (an unreadable header cannot rescue it).
                report.orphaned.append((pyc, f"no source: {sibling}"))
                continue
            co = _co_filename(pyc)
            if co is None:
                # Sibling exists but the bytecode cannot be parsed. CPython
                # will recompile it anyway; the cache is dead weight.
                report.stale.append((pyc, "<unreadable bytecode>"))
                continue
            co_path = Path(co)
            if not co_path.is_absolute():
                # Relative co_filename: recorded relative to wherever the
                # compiler ran (repo root in practice). Resolve against every
                # plausible anchor; an absolute-only check false-positives
                # "tests/x.py" into "tests/tests/x.py" and churn-deletes a
                # perfectly valid cache every session (measured 2026-10-12).
                candidates = (
                    sibling.parent / co_path,  # source dir
                    root / co_path,  # scan root (caller's tree)
                    _REPO_ROOT / co_path,  # repo root (py_compile from root)
                    Path.cwd() / co_path,  # compile-time cwd
                )
                if any(c.exists() for c in candidates):
                    report.kept += 1
                    continue
                report.stale.append((pyc, co))
                continue
            if not co_path.exists():
                report.stale.append((pyc, co))
            else:
                report.kept += 1
    return report


def _git_tracked(path: Path, root: Path) -> bool:
    """Whether git currently holds ``path`` (relative scan root ``root``).

    Defence in depth, ported from 81145e743: bytecode is derived, never
    tracked. Refusing to delete anything git holds means this tool cannot eat
    a tracked file even if the classification above were wrong.
    """
    try:
        rel = str(path.relative_to(root))
    except ValueError:
        rel = str(path)
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", rel],
        cwd=str(root),
        capture_output=True,
    )
    return result.returncode == 0


def _delete(items: list[tuple[Path, str]], root: Path, report: PurgeReport) -> None:
    for pyc, _ in items:
        if _git_tracked(pyc, root):
            report.skipped_tracked += 1
            print(f"  skipped (tracked): {pyc}", file=sys.stderr)
            continue
        try:
            pyc.unlink()
            report.deleted += 1
        except OSError as exc:
            print(f"  failed {pyc}: {exc}", file=sys.stderr)


def purge_stale_pyc(
    roots: list[Path] | None = None,
    *,
    apply: bool = False,
    verbose: bool = False,
) -> PurgeReport:
    """Classify (and with ``apply=True`` delete) stale/orphaned bytecode.

    Importable entry point — the pytest_sessionstart guard in
    tests/conftest.py calls this. Never raises: bytecode is derived cache, so
    any failure degrades to "kept", never to a broken test session.
    """
    try:
        scan_roots = [Path(r) for r in (roots or _DEFAULT_ROOTS)]
        report = classify(scan_roots)
        if apply:
            for root in scan_roots:
                _delete(
                    [item for item in report.removable if item[0].is_relative_to(root)],
                    root,
                    report,
                )
        if verbose:
            _print_report(report, scan_roots, applied=apply)
        return report
    except Exception as exc:  # noqa: BLE001 — guard hook must never break a session
        print(f"purge_stale_pyc: degraded ({exc!r})", file=sys.stderr)
        return PurgeReport()


def _print_report(report: PurgeReport, roots: list[Path], *, applied: bool) -> None:
    verb = "Deleted" if applied else "Would delete"
    print(f"Scanned {report.scanned} .pyc under {', '.join(str(r) for r in roots)}")
    print(
        f"  orphaned: {len(report.orphaned)}   stale: {len(report.stale)}"
        f"   kept: {report.kept}"
    )
    for label, items in (("orphaned", report.orphaned), ("stale", report.stale)):
        for pyc, detail in items[:5]:
            print(f"    [{label}] {pyc}\n        -> {detail}")
        if len(items) > 5:
            print(f"    [{label}] ... and {len(items) - 5} more")
    if applied:
        print(
            f"{verb} {report.deleted}; skipped tracked: {report.skipped_tracked}"
        )
    elif report.removable:
        print(f"{verb} {len(report.removable)} (dry-run; pass --apply to delete)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "roots",
        nargs="*",
        type=Path,
        help="Directories to scan (default: tests/ packages/ under the repo root)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete. Without this flag the run is a dry-run report.",
    )
    args = parser.parse_args()

    roots = [r.resolve() for r in args.roots] if args.roots else list(_DEFAULT_ROOTS)
    purge_stale_pyc(roots, apply=args.apply, verbose=True)
    return 0 if all(r.is_dir() for r in roots) else 1


if __name__ == "__main__":
    sys.exit(main())
