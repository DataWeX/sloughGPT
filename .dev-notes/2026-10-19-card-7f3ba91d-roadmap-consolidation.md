---
id: 7f3ba91d
title: "Consolidate the two ROADMAP.md files into one canonical roadmap with Now/Next/Later tracks and a v1 release definition"
status: done
board_id: Main
date: 2026-10-19
branch: docs/roadmap-restructure
tags: [docs, product, roadmap]
---

# Card 7f3ba91d — roadmap consolidation (2026-10-19)

Worktree `/home/mana/Documents/sloughgpt-roadmap`, branch `docs/roadmap-restructure`
(base origin/main `dbccf0ad9`). Doc edit only — no code, no installs.

## Problem (user-asked audit: "have we written a sufficient roadmap?")

Four defects, all verified:

1. **Two files, one name, contradicting each other.** Root `ROADMAP.md` (not
   registered in docs/INDEX.md) claims "Phase 7 Mobile (Future)" fully ✅;
   `docs/ROADMAP.md` says "Mobile: not started". `apps/README.md` shows
   `apps/mobile/` exists (scaffold, ~7.5G node_modules) — both claims wrong
   in opposite directions.
2. **Canonical roadmap is a victory log** — near/medium/new goals all struck
   Done; only training goals 15–19 + owned milestones 12–14 look forward.
3. **Product/UX track lives only in PRODUCT_ENGINEERING.md** (route map
   109→20 pages: 7 ⏳ targets; build order #5 UI flows, #7 journey gate) —
   unsequenced, unreleased, invisible from the roadmap.
4. **No release definition** + root velocity table frozen 2026-07-13, "How to
   Pick the Next Task" points at the fully-done Phase 6.

## Decisions taken (user, 2026-10-19)

- "Structure it well" → rewrite both files as agreed.
- **v1 = route map first**: ship the 7 ⏳ route-map page merges + build-order
  #5 (UX_FLOWS page restructures) + #7 (journeys green) BEFORE the training
  track ships as the release.
- Root `ROADMAP.md` → pointer stub (NOT deleted — needs approval to remove).
- Forward items **reference card IDs, never restate card status** (notes↔board
  lesson applied to docs; restated status drifts the moment the board moves).

## Scope guard

- Base content on **origin/main's** copy (newer truth-ups), NOT the root
  checkout's `fix/training-handlers-unify` copy (older — preflight proved it).
- Do NOT swallow PRODUCT_ENGINEERING.md (what/why) or UX_FLOWS.md (flows):
  reference them, don't restate.
- Do NOT renumber existing goals 1–20 — notes/cards cite them by number
  ("Roadmap #1", goal 17 gate).
- Direction block, "bootstrap era" lessons, two-track training split, owned
  milestones 12–14, deferred-Rust table: keep (user-approved as strong).
- Resolve the mobile contradiction with evidence, not by picking a side.

## Coordination

- No session owns ROADMAP files (preflight: both clean vs HEAD, no foreign
  hunks). Root checkout is `fix/training-handlers-unify` — untouched by this
  card; that branch's older docs/ROADMAP.md will conflict on its next rebase
  (its authors should take main's version).
- docs/INDEX.md + docs/README.md already register docs/ROADMAP.md — no index
  change needed (no NEW doc created).

## Acceptance

- Single canonical `docs/ROADMAP.md`: direction kept; Now/Next/Later × 3
  tracks (Product, Model, Platform); v1 release definition = route map first;
  mobile contradiction resolved; card IDs referenced.
- Root `ROADMAP.md` = pointer stub, no content loss (history stays in git).
- No test references ROADMAP files (grep tests/ → 0 hits) + docs/INDEX entry
  intact; gates run anyway (merge-when-green), PR merged with evidence here.

## EVIDENCE

- **PR**: #164 — "docs: one canonical roadmap — Now/Next/Later, v1 = route map first"
- **Merge commit**: `828638485b392d80de464adec61ed3f4b5c2c79e` (MERGED 2026-10-19T11:45:32Z, mergedBy iamtowbee, base main)
- **Branch**: `docs/roadmap-restructure`, base `origin/main` `dbccf0ad9`; commit `fd3972e98` touched EXACTLY `ROADMAP.md` + `docs/ROADMAP.md` (`git show --stat`: 2 files, +135/−200, no swept files).

### Gates
- `git status --porcelain`: only `ROADMAP.md` + `docs/ROADMAP.md` dirty — nothing else.
- `git diff --check`: clean (exit 0, no whitespace errors).
- **Shared-index gate**: `tests/test_shared_indexes.py` is ABSENT on base `dbccf0ad9` (branch tree) → could not run in the worktree (pytest exit 4 = file-not-found, NOT a pass). Intent verified directly instead: `docs/INDEX.md:13` and `docs/README.md:14` still register `docs/ROADMAP.md`; diff touches no index files. The harness itself is green (6 passed) where the file exists (local main checkout).
- **Sanity-read**: `docs/ROADMAP.md` contains "## How to read this roadmap", "## Now — v1 release: route map first", "### Deferred (potential Rust)". Root `ROADMAP.md` is a pointer stub — phase ledger retired (its single "Phase 7" hit is retirement-rationale prose, not a phase entry).

### CI (PR #164)
- **14 PASS**: Contract gate, E2E tests, Feature Tags, Frontend unit tests, Next.js build, Python (3.11), Python (3.12), TypeScript, test-root, test-sdk-py, test-sdk-ts, test-server, test-strui, test-web.
- **1 FAIL**: `lint` — `ruff format --check` on `packages/core-py/tests/test_slonet_brain.py:305` (a Python test file this docs-only PR does NOT touch). Byte-identical to `origin/main` `dbccf0ad9`; PRE-EXISTING and RED on `origin/main` (main run `39238828768` on sha `dbccf0ad9`: lint=failure).
- **1 pending at merge**: `test-core-py` (large 40–60 min suite). Also RED on `origin/main` at the same sha (run `39238828768` on `dbccf0ad9`: test-core-py=failure). Docs-only ⇒ my Python tree is byte-identical to `dbccf0ad9`, so this outcome is deterministic and pre-existing.
- **Merged under the docs-only exception**: both non-green checks shown red on `origin/main` at the exact base sha; every other check green. (Consistent with #162/#163, which also merged with red CI/CD.)

### Board note — card registration BLOCKED (infra, not this change)
- Card `7f3ba91d` was **never registered on the board**. This dev-note is a markdown file in the `~/.config/dev-notes/` fallback, but this repo's board is driven by the **project journal backend** (`.dev-notes/notes.journal.jsonl`, backend=`mogdb`), which does not read fallback markdown.
- `note update 7f3ba91d --status wip` → `Note not found: 7f3ba91d` (exit 1); `board show 7f3ba91d` → `Card not found: 7f3ba91d` (exit 1); `sync` produced no card; grep `.kanban/board.jsonl` → 0 matches.
- Consequently there is **no board column** for this card — it does not exist on the board. The frontmatter here was set to `done` for the human record, but the CLI/board path (`note update` → `board move` → `sync`) is blocked by the note-store mismatch and was intentionally NOT forced (forcing it would write a second, id-mismatched source into the shared journal).
