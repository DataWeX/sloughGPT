---
status: done
priority: high
tags: [tooling, backup, kanban]
board: e374b40d-3951-4207-9425-1f0cd0bedbab
landed: a1ebb7f2d (main/api-std), bb9604ea5 (feat/logs-unification)
hash: cbd34d65e3c9d867a4fd36dc3f3c6509031ea93b55274df78842ea61257a97c7
---

# Card 095 — Catastrophic-backup: journal + boards + code offsite-safe

## What

`scripts/backup_critical.sh` — extinction-level-event protection:

- **local leg**: dated `slog-backup-<ts>.tgz` (journal incl. git history,
  both `.kanban/board.jsonl` worktrees, optional full `git bundle`) +
  sha256 manifest, retention (keep 10), `--verify` re-hash → survives
  alteration/ransomware via dated snapshots.
- **offsite leg**: `git push origin --all --tags` (code) + journal repo
  commit/push to **private** `github.com/DataWeX/dev-notes` (board
  snapshots refreshed in `snapshots/` each run) → survives fire/disk death.
- modes: default full run, `--local` (no push), `--full` (add bundle),
  `--verify [file|all]`.

## Exposure found (2026-09-30)

- `~/.config/dev-notes` (108 notes) was **not a git repo** — one disk only.
- `.kanban/board.jsonl` untracked in the code repo.
- api-std `main` **523 commits ahead** of `origin/main` (stale remote).

## Verification (all green)

- `--verify OK` on both archives; **restore test**: extracted tgz →
  `diff -r` journal + board **byte-identical**.
- **bundle test**: `slog-backup-20260930T134127Z.tgz` (262M, full
  `--all` bundle) → `git bundle verify` "complete history" → mirror clone
  = **230 refs, `main` = bbf46e7c6, card-092 history present**.
- **offsite**: GitHub `DataWeX/sloughGPT` now has **209 branches + 3 tags**,
  `main` at current head (was 523 behind); pre-push secret scan of
  `origin/main..main`: **0 real secrets** (filename hits all "tokenizer").
- **journal**: private repo `DataWeX/dev-notes`, local HEAD == remote
  (`fae395c`, incl. both board snapshots).
- lands: `a1ebb7f2d` (main/api-std), `bb9604ea5` (feat/logs-unification);
  their staged set byte-intact after landing.

## Ops

- Run `./scripts/backup_critical.sh` after sessions (or wire a cron);
  `--verify` is read-only and CI-safe.
