# 073 — Product docs truth-up (PRODUCT_ENGINEERING / ROADMAP / FEATURES / INDEX)

- **Sprint**: 2026-09 (tranche: doc↔code seams, after 071 routers/testing)
- **Status**: in-progress
- **gh**: (local)
- **Branch**: feat/routers-doc-truthup (same worktree sloughgpt-routers-doc, seam 4 of 071's queue) — OR new branch? folded into 073 card for tracking.
- **Body**: Audit docs/PRODUCT_ENGINEERING.md (260), docs/ROADMAP.md (96), docs/FEATURES.md (291), docs/INDEX.md (75) against actual code. Same pattern as seams 1-3: find stale claims (dead-code table vs build-order ✅, voice/kb/training "bad examples" vs fixed routers, "58 routers", "324 files/3048 tests", page/prefix tables), truth-up, add contract guard (backticked paths exist, feature-table API prefixes ⊆ live routes, frontend pages exist).
- **Blocked by**: 071 (shares branch), 067 (merge queue)

## Findings (seam-4 audit, 4 docs / 722 lines)

Stale/false claims fixed (commit `41fd73da6` on feat/routers-doc-truthup):
- PRODUCT_ENGINEERING: Tools frontend page /tools (dir removed); headline
  '85+ pages, 58 routers' (real: 109 pages, 56 routers/*.py); entire 'Bad
  examples' section (voice/kb/tokenizer/training bypasses — all repaired;
  kb.py 1399 lines -> no count claimed; training/router.py uses facades with
  allowlisted training/ package exception: 72 _internal imports live there
  under scatter-gate); dead-code table (4 of 5 rows already removed, api_keys
  still open); phoneme '47 components' + tokenizer '9 tabs' de-claimed;
  consciousness 27 -> 25 sub-pages.
- ROADMAP: 'Web: 32 pages' -> 109; 'Voyager (201 tests, 7 backends)' ->
  packages/avion (225 tests, 7 backends; voyager = stub, goal marked
  Superseded); '324 files / 3048 tests' -> excluded-file fact + npm test
  pointer (live include-glob count: 836 of 838 test files).
- FEATURES: consciousness '26 pages' -> 25 (page tables match live dirs
  exactly); CLI table rewritten (Click-era `datasets`/`autotrain`/
  `feedback-export`/`apps/cli/cli.py`/`train --dataset` all gone; current =
  ./sloughgpt dataset|feedback|train groups, ~44 top-level commands);
  Tools Pages section (7 pages) -> removal note; self-reported coverage
  table -> TESTING.md pointer.
- INDEX.md: verified clean (all 32 doc references exist).

Verified-true claims left alone: UX_FLOWS 13 journeys; REPL 40+ commands
(58 _cmd_ handlers); avion 7 backends; dataset import endpoints LIVE;
FEATURES consciousness page tables (25/25 exist); core feature-table API
prefixes all live.

## Guard: tests/contract/test_product_docs.py (6 tests)

1. backticked repo paths exist (negative-context lines skipped: removed/
   deleted/superseded/stub/formerly...)
2. forbidden stale strings per doc (58 routers, 32 pages, 201 tests,
   324 files, 26 pages, Click-era CLI literals, | 100% | tables...)
3. feature-table API prefixes (6-col numeric rows) have live routes
   (probe via test_routers_doc._live_routes, loaded by path) + page cells
   exist under app/(app)
4. headline 'N frontend pages, M routers' == live counts (exact)
5. consciousness counts in both docs == dirs with page.tsx (25)
6. facade claim number == routers/*.py count; routers/ has 0 _internal
   imports (reimport reintroduction canary)

Mutation-verified: stale headline reverted -> caught; dead path injected ->
caught.

## Gates
- pytest tests/contract/: **29 passed in 10.2s** (23 prior + 6 new)
- ruff check + format: clean on all contract tests
- TESTING.md contract-suite count updated 23 -> 29 (self-counting guard)

## Report log
- 05:5x — committed 41fd73da6; merge watcher (pids 132093/132116, 24h
  windows) still parked on api-std dirty=29, main@f1ecf8a6f.
