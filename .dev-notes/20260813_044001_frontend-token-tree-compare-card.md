---
id: 20260813_044001_frontend-token-tree-compare-card
title: Frontend token-tree compare card
status: done
tags: frontend,tokenizer,compare
created: 2026-08-13T04:40:01.293067+00:00
---

Frontend token-tree compare card

Completed the compare feature end-to-end in the frontend (API + CLI were already wired).

Controller (apps/web/lib/token-tree-controller.ts): added compare(a, b, top_k=10) hitting POST /token-tree/compare; added CompareResult + CompareSide types matching the API response ({a, b, shared_tokens, only_a/only_b_tokens, shared_merges, only_a/only_b_merges, shared/only_*_examples}). Removed duplicate MatrixEnergyToken interface. Exported CompareResult, CompareSide, SavedTree, MatrixSummary, VocabPage from controllers.ts barrel. 2 new tests (17 total).

Card (apps/web/components/tokenizer/TokenTreeCompareCard.tsx): two strui Selects for tree A/B from listSaved(), Compare button disabled until two distinct trees picked, result panel with per-side vocab chips, shared/only-A/only-B token counts, merge rule chips, and top-frequency example token lists. Empty state when <2 trees, retry state on fetch failure, error toast on compare failure, refreshKey refetch. Wired into tokenizer page after TokenTreeLineageCard.

Tests: 8 new card tests + 2 controller tests; tokenizer suite 142 tests pass; tsc exit 0.