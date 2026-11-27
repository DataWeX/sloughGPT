#!/usr/bin/env python3
"""Lane-claims advisory — warn when your dirty paths belong to another session.

Policy: AGENTS.md "Lane claims" + coordination-workflow skill.

Failure this prevents: 2026-11-26 voice lane — a parallel session worked the
same checkout; `git status` showed files that looked free (or silently
reverted edits at 09:49 and 10:38) with no way to tell who was there. Claims
(`grib claim add`) record presence in the canonical store; this script is the
read-side correlation: which active claims (a) own THIS worktree outright, or
(b) name card paths that match my uncommitted files.

Advisory by default: always exit 0 (prints WARN lines). --strict exits 1 —
for wiring as a required check later, same ratchet as check_surface_scope.py.

Usage:
  scripts/check-lane-claims.py              # correlate claims vs `git status`
  scripts/check-lane-claims.py --strict     # exit 1 on any conflict
  scripts/check-lane-claims.py --json       # machine-readable findings
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "packages" / "grib" / "src"))

from grib.claims import (
    default_session,  # noqa: E402
    detect_git,  # noqa: E402
    fold,  # noqa: E402
)

# Path-ish tokens from card descriptions: known top-level dirs with a dot
# (file-ish) or a wildcard segment. Deliberately conservative — an advisory
# that cries wolf gets ignored.
_PATH_TOKEN = re.compile(
    r"(?:apps|domain|packages|scripts|tests|docs|\.opencode)/[\w./-]*"
    r"(?:[\w-]|\*)"
)


def dirty_paths(cwd: Path) -> list[str]:
    """Untracked + modified paths relative to the repo root (porcelain v1)."""
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=cwd,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []
    paths: list[str] = []
    for line in proc.stdout.splitlines():
        if len(line) < 4:
            continue
        # "XY path" / "XY orig -> path" (renames); strip the status prefix.
        path = line[3:].split(" -> ")[-1].strip().strip('"')
        if path:
            paths.append(path)
    return paths


def card_paths(board_dir: Path, card_id: str | None) -> list[str]:
    """Path-like tokens named in the claimed card's title+description."""
    if not card_id:
        return []
    try:
        from grib.store import PlannerStore

        store = PlannerStore(board_dir=board_dir, notes_dir=board_dir)
        card = store.get_card(card_id)
    except Exception:  # noqa: BLE001 — advisory must not crash on store shape
        return []
    if card is None:
        return []
    text = f"{card.title}\n{card.description}"
    return _PATH_TOKEN.findall(text)


def matches(dirty: str, token: str) -> bool:
    token = token.rstrip("/")
    if "*" in token:
        pattern = re.compile(re.escape(token).replace(r"\*", "[^/]*"))
        return bool(pattern.match(dirty))
    return dirty == token or dirty.startswith(token + "/")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--strict", action="store_true", help="exit 1 on conflict")
    ap.add_argument("--json", action="store_true", help="machine-readable findings")
    ap.add_argument("--session", default=None, help="override own session id")
    ap.add_argument("--board-dir", default=None, help="override canonical board dir")
    args = ap.parse_args(argv)

    from grib import config

    board_dir = Path(args.board_dir) if args.board_dir else config.default_board_dir()
    me = args.session or default_session()
    here, _branch = detect_git()
    dirty = dirty_paths(Path(here) if here else Path.cwd())

    findings: list[dict] = []
    for claim in fold(board_dir):
        if claim["status"] != "active" or claim["session"] == me:
            continue
        # (a) the whole checkout is theirs
        if here and claim["worktree"] and Path(claim["worktree"]).resolve() == Path(here).resolve():
            findings.append(
                {
                    "kind": "worktree-owned",
                    "session": claim["session"],
                    "card_id": claim["card_id"],
                    "age_min": round(claim["age_min"]),
                }
            )
            continue
        # (b) my dirty files match paths their card claims
        tokens = card_paths(board_dir, claim["card_id"])
        hits = sorted({p for p in dirty for t in tokens if matches(p, t)})
        if hits:
            findings.append(
                {
                    "kind": "dirty-path-overlap",
                    "session": claim["session"],
                    "card_id": claim["card_id"],
                    "age_min": round(claim["age_min"]),
                    "paths": hits[:10],
                }
            )

    if args.json:
        print(json.dumps({"session": me, "worktree": here, "findings": findings}, indent=2))
    elif not findings:
        print(f"lane-claims: ok — no foreign active claim touches this checkout ({me})")
    else:
        for f in findings:
            if f["kind"] == "worktree-owned":
                print(
                    f"WARN: worktree claimed by {f['session']} "
                    f"(card={f['card_id'] or '-'}, {f['age_min']}m ago) — you are in THEIR checkout"
                )
            else:
                print(
                    f"WARN: {len(f['paths'])} dirty path(s) match card {f['card_id'] or '-'} "
                    f"claimed by {f['session']} ({f['age_min']}m ago): " + ", ".join(f["paths"])
                )
        print(f"lane-claims: {len(findings)} conflict(s); release/refresh the claim or move lanes")

    if findings and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
