#!/usr/bin/env python3
"""
CI alarm: the local kanban sync producer must be alive.

The canonical store lives outside CI (see AGENTS.md architecture), so
``scripts/kanban_sync.sh`` (kanban-sync.timer) mirrors it into
``.kanban/board.jsonl`` + ``.kanban/manifest.json``. If that timer dies,
board history on GitHub goes stale silently. This check reddens the build
instead: manifest missing or ``exported_at`` older than the budget.

Reads the manifest from ``origin/main``, NOT from the checked-out tree: the
producer pushes to main and only main, so on any feature branch the working
copy is stale *by design* and reddening it is a false positive (observed —
every branch CI run printed "kanban-sync.timer looks dead"). ``--ref`` is the
source of truth; ``--manifest`` overrides with a literal file (what the gate
tests use).

Stdlib only, no app imports — CI runs it on a stock python3.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_IN_TREE = ".kanban/manifest.json"
DEFAULT_REF = "origin/main"
DEFAULT_MANIFEST = REPO_ROOT / MANIFEST_IN_TREE
DEFAULT_MAX_AGE_MINUTES = 120


def run_git(args: list[str], cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess[str]:
    """One git call, kept tiny so the gate tests can inject their own."""
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=60)


def read_from_ref(ref: str) -> tuple[str | None, str]:
    """``origin/main``'s manifest text, fetching once if the ref is absent.

    Returns ``(text, reason)`` — exactly one of them is set. A CI checkout
    with depth 1 may not carry the ref at all, so a missing ref is retried
    against the remote before giving up.
    """
    result = run_git(["show", f"{ref}:{MANIFEST_IN_TREE}"])
    if result.returncode == 0:
        return result.stdout, ""
    first = result.stderr.strip() or f"{ref} does not resolve"

    fetch = run_git(["fetch", "--no-tags", "origin", "main"])
    if fetch.returncode != 0:
        return None, f"{first} (fetch failed: {fetch.stderr.strip() or 'no origin'})"

    result = run_git(["show", f"{ref}:{MANIFEST_IN_TREE}"])
    if result.returncode == 0:
        return result.stdout, ""
    return None, f"{first} (still unreadable after fetch)"


def load_manifest(args: argparse.Namespace) -> tuple[dict | None, str, str]:
    """(manifest, source label, error). The source label is printed so a
    verdict can always be attributed to the tree it came from."""
    if args.manifest_explicit:
        if not args.manifest.is_file():
            return (
                None,
                str(args.manifest),
                (
                    f"no {args.manifest} — the sync producer never ran "
                    f"(scripts/kanban_sync.sh / kanban-sync.timer)"
                ),
            )
        raw, source = args.manifest.read_text(encoding="utf-8"), str(args.manifest)
    else:
        raw, problem = read_from_ref(args.ref)
        source = f"{args.ref}:{MANIFEST_IN_TREE}"
        if raw is None:
            return None, source, problem
    try:
        return json.loads(raw), source, ""
    except json.JSONDecodeError as exc:
        return None, source, f"unreadable manifest {source}: {exc}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--ref", default=DEFAULT_REF, help="git ref to read the manifest from")
    parser.add_argument("--max-age-minutes", type=int, default=DEFAULT_MAX_AGE_MINUTES)
    args = parser.parse_args(argv)
    # An explicitly passed --manifest means "judge THIS file" (the gate tests,
    # and any operator replaying a copy); the bare default means "judge main".
    args.manifest_explicit = "--manifest" in (argv if argv is not None else sys.argv[1:])

    manifest, source, problem = load_manifest(args)
    if manifest is None:
        if args.manifest_explicit:
            print(f"FAIL: {problem}")
            return 1
        # Reading main is a precondition we do not control (shallow CI
        # checkout, no origin). Fail OPEN, loudly — same policy as
        # scripts/check_board_chain.py, because "cannot verify" must never
        # be reported as "the producer is dead".
        print(f"WARN: cannot verify kanban freshness from {source}: {problem}")
        return 0

    try:
        exported = datetime.fromisoformat(manifest["exported_at"])
    except (KeyError, ValueError) as exc:
        print(f"FAIL: unreadable manifest {source}: {exc}")
        return 1
    if exported.tzinfo is None:
        exported = exported.replace(tzinfo=UTC)

    age = datetime.now(UTC) - exported
    limit = timedelta(minutes=args.max_age_minutes)
    if age > limit:
        print(
            f"FAIL: kanban export on {source} is stale: age={age} > {limit} — kanban-sync.timer looks dead"
        )
        return 1

    print(
        f"ok: kanban export on {source} is {age} old "
        f"({manifest.get('card_count', '?')} cards, "
        f"{manifest.get('note_count', '?')} notes)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
