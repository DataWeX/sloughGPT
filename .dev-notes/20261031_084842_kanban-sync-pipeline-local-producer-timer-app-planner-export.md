---
id: 20261031_084842_kanban-sync-pipeline-local-producer-timer-app-planner-export
status: done
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

- **Landed on main** (`feat/kanban-sync-pipeline` → `HEAD:main`):
  `a4431cb4e` (10 files +523: export module+tests, CLI export, sync script,
  units, freshness script+tests, ci.yml step, AGENT_SYNC entry) + heartbeat
  fix `892eafc37`. One push-race rebase onto `654f32e6e` mid-landing,
  zero conflicts.
- **First live run** `2fae706ff`: 230 paths mirrored+pushed; GitHub
  verified (`git ls-tree origin/main .kanban/manifest.json` + manifest
  content: 228 notes / 12 cards / board sha).
- **Timer live**: `kanban-sync.timer` enabled + active since 10:50:33;
  manual `systemctl --user start kanban-sync.service` → exit 0.
- **Idempotency live-proven post-fix**: plain script run = "up to date",
  timer run 11:08:57 = "up to date" (zero commits), `KANBAN_SYNC_HEARTBEAT=0`
  = gate opens → "pushed (1 paths)".
- **Verifier**: local freshness `ok: kanban export 0:07 old (12 cards,
  228 notes)`; `check_board_chain.py` ok; working tree clean at `892eafc37`.
- Note: pre-fix timer commits (`247d66d5a`, `02d068e61`, `2afd6b599`) are
  the `exported_at` churn this card's heartbeat fix ended — kept as history,
  no cleanup.

## Pitfalls

- `.gitignore:277` `.dev-notes/*.md` — producer `git add -f` (mirror is
  authoritative; ordinary sessions keep never committing stale copies).
- Write tool truncated `cli.py` tail once mid-edit (disk was at 0 bytes) —
  disk pressure again: repaired, diff verified `+35/-0` pure-add.
- `main` checked out by worktree fe08f8/jolly-wizard → push `HEAD:main` only.
