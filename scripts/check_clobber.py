#!/usr/bin/env python3
"""Advisory clobber detector — flag commits that tear out fix lineage.

WHY THIS EXISTS (card cf308a33 follow-up)
-----------------------------------------
A single-parent "converge X to main — was N lines stale" commit wholesale-
overwrites a tree from a stale snapshot and silently reverts fixes that landed
in the interim (074475d26 avion, 81616a6f7 EmbeddingConfig, 0a29d5c6e monitor).
tests/test_regression_locks.py locks the KNOWN fixes; this hunts the UNKNOWN
ones.

WHY IT IS ADVISORY, NOT A BLOCKER
---------------------------------
Judging "this line was a fix" needs human sense a heuristic cannot make, so a
hard gate would fail CI on false positives. This prints candidates to eyeball.
The real guard is the regression-lock test file.

THE SIGNAL (measured over 1500 commits, not guessed)
----------------------------------------------------
Width does NOT work: a stale-snapshot converge is dominated by INSERTIONS that
restore older lines (the avion clobber is only 4% deletions), so a churn/width
threshold flagged 235/1200 commits (20%) — including legitimate refactors.

The precise signal is REVERSAL OF DOCUMENTED LINEAGE: a fix leaves a comment
naming the offending sha (e.g. "# ef58a0c9c passed a positional %s arg ..."),
and a clobber deletes that comment along with the fixed line. So: flag commits
that delete a line which references a git sha and reads as a fix/revert note.

Measured result — the avion clobber IS caught, and noise drops to 0.67%
(10/1500) once board/journal paths are excluded (chore(kanban) commits edit
card prose that cites shas — those are not code lineage). All 9 remaining hits
are wide, legitimate refactors: acceptable for an advisory second-look.

It reads only the commits under examination, so it works on shallow clones —
it never walks deep history (the reason a git-history detector was rejected).
"""

from __future__ import annotations

import re
import subprocess
import sys

# A line is "lineage" if it names a git sha AND reads as a fix/revert note.
_SHA_RE = re.compile(r"\b[0-9a-f]{7,10}\b")
_LINEAGE_RE = re.compile(r"#|fix|clobber|revert|regress|workaround", re.I)

# Board/journal text cites shas as card prose — not code lineage. Exclude them.
_SKIP_PREFIXES = (".kanban/", ".dev-notes/")

DEFAULT_WINDOW = 1500


def _git(repo: str, *args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=repo).stdout


def _is_merge(repo: str, sha: str) -> bool:
    """Merges legitimately reconcile two lines — not the single-parent shape."""
    return len(_git(repo, "show", "-s", "--format=%P", sha).split()) > 1


def _deleted_lineage_lines(repo: str, sha: str) -> list[str]:
    """Deleted lines that reference a sha and read as fix/revert lineage.

    Board/journal paths are skipped — their sha text is card prose.
    """
    hits: list[str] = []
    path = ""
    for line in _git(repo, "show", "--format=", sha).splitlines():
        if line.startswith("+++ b/"):
            path = line[6:]
            continue
        if not line.startswith("-") or line.startswith("---"):
            continue
        if path.startswith(_SKIP_PREFIXES):
            continue
        body = line[1:]
        if _SHA_RE.search(body) and _LINEAGE_RE.search(body):
            hits.append(body.strip())
    return hits


def scan(repo: str = ".", window: int = DEFAULT_WINDOW) -> list[dict]:
    """Return clobber candidates: non-merge commits that delete fix lineage."""
    shas = _git(repo, "rev-list", "--max-count", str(window), "HEAD").split()
    candidates: list[dict] = []
    for sha in shas:
        if _is_merge(repo, sha):
            continue
        lineage = _deleted_lineage_lines(repo, sha)
        if lineage:
            candidates.append(
                {
                    "sha": sha,
                    "lineage": lineage,
                    "message": _git(repo, "show", "-s", "--format=%s", sha).strip(),
                }
            )
    return candidates


def main(argv: list[str]) -> int:
    window = int(argv[0]) if argv else DEFAULT_WINDOW
    cands = scan(window=window)

    if not cands:
        print(f"check_clobber: no fix-lineage deletions in last {window} commits")
        return 0

    print(
        f"check_clobber: {len(cands)}/{window} commit(s) deleted fix-lineage "
        f"comments in the last {window} commits\n"
    )
    for c in cands:
        print(f"  {c['sha'][:10]}  {c['message'][:80]}")
        for ln in c["lineage"][:2]:
            print(f"        - {ln[:88]}")
    print(
        "\n  Advisory only. A wide sweep may have reverted an interim fix whose\n"
        "  lineage comment it also deleted. For each: confirm it was not built\n"
        "  from a stale snapshot and that the fix it touched still holds.\n"
        "  Known fixes are pinned in tests/test_regression_locks.py."
    )
    return 0  # never blocks — human judgement decides


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
