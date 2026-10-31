---
id: 20261031_084842_kanban-sync-pipeline-local-producer-timer-app-planner-export
status: in_progress
board: 20261031_084842_kanban-sync-pipeline-local-producer-timer-app-planner-export
title: "kanban sync pipeline: local producer timer + app-planner export + CI freshness alarm"
tags: [kanban, infra, ci, systemd, app-planner]
sprint: domain-hygiene
gh: null
---
# Kanban → GitHub sync pipeline (user: "juste")

## Why

Canonical state flipped out of the repo (2026-10-27): `~/.config/kanban/board.jsonl`
+ `~/.config/dev-notes/*.md` are invisible to CI and GitHub. Board history on
GitHub went stale silently with nothing watching. Split:

- **Producer** (local, single writer): mirror canonical → repo copies + push.
- **Verifier** (CI): chain check (already there) + **freshness alarm** (new) —
  a dead producer reddens every PR instead of drifting silently.

## Design (approved)

1. `app_planner.export.export()` — live mirror:
   canonical `board.jsonl` → `<repo>/.kanban/board.jsonl` (NOT
   `.kanban/snapshots/` — `snapshot()` is dated-append, wrong shape for a
   5-min timer); canonical `*.md` → `<repo>/.dev-notes/*.md` byte-exact,
   prunes repo notes canonical dropped; writes `.kanban/manifest.json`
   `{exported_at, note_count, card_count, board_sha256, source}`.
   Refusals (never silent-empty): missing canonical board, missing notes dir.
2. CLI subcommand `app_planner.cli export --repo <root>` (dispatch entry next
   to `snapshot`); no git operations inside export.
3. `scripts/kanban_sync.sh` — flock single-writer; ONE detached worktree
   `~/.cache/kanban-sync-wt` pinned to origin/main (`main` itself is checked
   out by the live jolly-wizard worktree; detached head also isolates every
   session tree from this writer); loop: fetch+reset → export → pathspec
   add/commit → push, on rejection re-export fresh and retry ×3; idempotent
   (no canonical change → no commit → exit 0).
4. `infra/systemd-user/kanban-sync.{service,timer}` — `OnUnitInactiveSec=5min`,
   `Persistent=true` (sleep catch-up; budget 120min).
5. `scripts/check_kanban_freshness.py` (stdlib) + step in `ci.yml`
   `board-integrity` job → `--max-age-minutes 120`.

## Gates

- `packages/app-planner/tests/` **331 passed** (incl. 7 new export tests:
  byte mirror, manifest sha/counts, prune-keeps-legacy, both refusals).
- `tests/test_check_kanban_freshness.py` **8 passed** (+ board-chain 15).
- ruff check + format clean; `bash -n`; ci.yml parses.

## Evidence

- (to fill: first live run + GitHub verification + timer active)

## Pitfalls

- `.gitignore:277` `.dev-notes/*.md` — producer `git add -f` (mirror is
  authoritative; ordinary sessions keep never committing stale copies).
- Write tool truncated `cli.py` tail once mid-edit (disk was at 0 bytes) —
  disk pressure again: repaired, diff verified `+35/-0` pure-add.
- `main` checked out by worktree fe08f8/jolly-wizard → push `HEAD:main` only.
