#!/usr/bin/env bash
# Kanban -> GitHub sync — producer half (card 20261031_084842).
#
# Mirrors the canonical store (~/.config/{kanban,dev-notes}) into a repo
# copy and pushes it, so GitHub history tracks board + notes without any
# session needing to remember:
#
#   1. single-writer lock            (flock — timer overlap or manual runs)
#   2. ONE detached worktree         (~/.cache/kanban-sync-wt pinned to
#                                     origin/main; `main` itself is checked
#                                     out by a live session worktree, and a
#                                     detached head also isolates every
#                                     session's tree from this writer)
#   3. export -> commit -> push loop (push races are a known event here:
#                                     on rejection re-export from fresh
#                                     origin/main and retry, x3)
#
# Idempotent: no canonical change -> no commit, no push, exit 0.
# Run manually or via kanban-sync.timer (infra/systemd-user/).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WT="${KANBAN_SYNC_WT:-$HOME/.cache/kanban-sync-wt}"
LOCK="${XDG_RUNTIME_DIR:-/tmp}/kanban-sync.lock"
ATTEMPTS="${KANBAN_SYNC_ATTEMPTS:-3}"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "kanban-sync: another run holds the lock — skipping"
  exit 0
fi

# The export worktree must be a real worktree of this repo; never rm -rf our
# way out of doubt — a broken worktree is a loud failure, not a cleanup job.
if [ ! -e "$WT/.git" ]; then
  if [ -e "$WT" ]; then
    echo "kanban-sync: $WT exists but is not a git worktree — refusing" >&2
    exit 1
  fi
  git -C "$REPO_ROOT" fetch origin main --quiet
  git -C "$REPO_ROOT" worktree add --detach "$WT" origin/main --quiet
fi

attempt=1
while [ "$attempt" -le "$ATTEMPTS" ]; do
  git -C "$REPO_ROOT" fetch origin main --quiet
  if ! git -C "$WT" reset --hard origin/main --quiet; then
    echo "kanban-sync: worktree at $WT cannot reset to origin/main" >&2
    exit 1
  fi

  # Producer: mirror canonical -> .kanban/board.jsonl + .dev-notes/ + manifest.
  PYTHONPATH="$WT/packages/app-planner/src" \
    "$WT/scripts/python" -m app_planner.cli export --repo "$WT" >/dev/null

  # -f: .dev-notes/*.md stays gitignored (.gitignore:277) so ordinary sessions
  # never commit stale copies; the producer force-adds the canonical mirror.
  # Once tracked, later `git add` runs update them normally (prunes stage as
  # deletions too). Pathspec-limited add+commit: never sweeps other files.
  git -C "$WT" add -f -- .kanban/board.jsonl .dev-notes

  # manifest.json carries exported_at, which export() refreshes EVERY run —
  # committing it unconditionally floods main with timestamp-only noise
  # (observed: two 1-path commits within 60s on the first live runs).
  # Commit it when the mirror content moved, OR when the last committed
  # heartbeat is older than KANBAN_SYNC_HEARTBEAT (default 1h; CI's freshness
  # budget is 120min, so a live timer never reddens).
  if ! git -C "$WT" diff --cached --quiet; then
    git -C "$WT" add -f -- .kanban/manifest.json
  else
    last="$(git -C "$WT" log -1 --format=%ct -- .kanban/manifest.json)"
    now="$(date +%s)"
    if [ $((now - ${last:-0})) -ge "${KANBAN_SYNC_HEARTBEAT:-3600}" ]; then
      git -C "$WT" add -f -- .kanban/manifest.json
    fi
  fi

  if git -C "$WT" diff --cached --quiet; then
    echo "kanban-sync: up to date"
    exit 0
  fi

  changed="$(git -C "$WT" diff --cached --name-only | wc -l)"
  git -C "$WT" commit --quiet -m \
    "kanban: sync $(date -u +%Y-%m-%dT%H:%M:%SZ) (${changed} paths)"

  if git -C "$WT" push origin HEAD:main --quiet; then
    echo "kanban-sync: pushed (${changed} paths)"
    exit 0
  fi

  echo "kanban-sync: push rejected (origin moved) — retry ${attempt}/${ATTEMPTS}"
  attempt=$((attempt + 1))
done

echo "kanban-sync: gave up after ${ATTEMPTS} attempts" >&2
exit 1
