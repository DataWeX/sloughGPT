"""Lane claims — append-only session/worktree presence in the canonical store.

A claim answers "which session is working which worktree/card RIGHT NOW",
which `git status` heuristics cannot (a clean-but-owned checkout, a branch
switched mid-task, or a parallel session reverting edits are all invisible —
incidents 2026-11-26, card 20261126_102530).

Design:
- **Append-only events** in ``<board_dir>/claims.jsonl`` (canonical store,
  never a repo file — worktrees fork repo-local state). No read-modify-write:
  O_APPEND single-line writes are atomic on POSIX, so concurrent sessions
  cannot tear the file.
- **Fold, don't mutate**: current state per session = its latest event;
  a ``release`` carries the session's last claim fields forward. Expiry is
  computed at read time from the heartbeat age vs the claim's TTL — there is
  no sweeper process to strand on a restart.
- **Notification = read-time visibility** (no push bus): sessions read claims
  when they read the queue and refresh their own claim on the same pass.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

DEFAULT_TTL_MIN = 120


def claim_path(board_dir: Path) -> Path:
    """Path of the claims event log for a board directory."""
    return Path(board_dir) / "claims.jsonl"


def default_session() -> str:
    """Session identity: $OPENCODE_SESSION_ID, else a host/pid degenerate."""
    sid = os.environ.get("OPENCODE_SESSION_ID")
    if sid:
        return sid
    user = os.environ.get("USER") or os.environ.get("USERNAME") or "unknown"
    return f"{user}@{socket.gethostname()}:{os.getpid()}"


def detect_git(cwd: Path | None = None) -> tuple[str | None, str | None]:
    """(worktree root, current branch) for *cwd*; (None, None) outside git."""

    def _run(*argv: str) -> str | None:
        try:
            proc = subprocess.run(
                argv, capture_output=True, text=True, timeout=10, cwd=cwd, check=False
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if proc.returncode != 0:
            return None
        return proc.stdout.strip() or None

    root = _run("git", "rev-parse", "--show-toplevel")
    if root is None:
        return None, None
    return root, _run("git", "branch", "--show-current")


def append_event(
    board_dir: Path,
    *,
    event: str,
    session: str,
    card_id: str | None = None,
    worktree: str | None = None,
    branch: str | None = None,
    ttl_min: int = DEFAULT_TTL_MIN,
    ts: float | None = None,
) -> dict[str, Any]:
    """Append one claim/release event (single atomic line, never rewritten)."""
    row: dict[str, Any] = {
        "ts": time.time() if ts is None else ts,
        "event": event,
        "session": session,
        "card_id": card_id,
        "worktree": worktree,
        "branch": branch,
        "ttl_min": ttl_min,
    }
    path = claim_path(board_dir)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    return row


def load_events(board_dir: Path) -> list[dict[str, Any]]:
    # Accept either the board directory or the claims file itself (callers
    # and tests pass both shapes).
    path = Path(board_dir)
    if not path.is_dir():
        path = path if path.exists() or path.suffix == ".jsonl" else claim_path(path)
    if path.is_dir():
        path = claim_path(path)
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a torn line must never take the registry down
    return events


def fold(board_dir: Path, *, now: float | None = None) -> list[dict[str, Any]]:
    """Current claim per session: latest event wins, expiry computed at read.

    Statuses: ``active`` (heartbeat within TTL), ``expired`` (stale — treat
    as free), ``released`` (explicitly given up).
    """
    now = time.time() if now is None else now
    latest: dict[str, dict[str, Any]] = {}
    last_claim: dict[str, dict[str, Any]] = {}
    for ev in load_events(board_dir):
        session = ev.get("session")
        if not session:
            continue
        if ev.get("event") == "claim":
            last_claim[session] = ev
        latest[session] = ev

    claims: list[dict[str, Any]] = []
    for session, ev in latest.items():
        prior = last_claim.get(session, {})
        age_min = (now - float(ev.get("ts", 0.0))) / 60.0
        ttl_min = int(ev.get("ttl_min") or prior.get("ttl_min") or DEFAULT_TTL_MIN)
        if ev.get("event") == "release":
            status = "released"
        elif age_min > ttl_min:
            status = "expired"
        else:
            status = "active"
        claims.append(
            {
                "session": session,
                "card_id": ev.get("card_id") or prior.get("card_id"),
                "worktree": ev.get("worktree") or prior.get("worktree"),
                "branch": ev.get("branch") or prior.get("branch"),
                "ts": ev.get("ts"),
                "age_min": age_min,
                "ttl_min": ttl_min,
                "status": status,
            }
        )
    claims.sort(key=lambda c: float(c.get("ts") or 0.0), reverse=True)
    return claims
