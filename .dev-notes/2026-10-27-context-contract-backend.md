---
title: "Context contract: ContextPart + budgeted selection + one renderer (backend half)"
status: done
board: b46de5eb-e080-46f9-9a72-744f9834d7ab
date: 2026-10-27
tags: [backend, inference, rag, context, cognition]


# Context contract — backend half

## Problem (audit 2026-10-27)

Input construction destroys provenance at every hop:

- Producers hold score/rank/locator and discard them in-function:
  `rag.py:685` (context = "\n\n".join), `math_rag.py:600` (ScoredChunk → join),
  `knowledge_augmenter.py:196` (content[:300], score dropped at the gate).
- Three blind char-truncation policies coexist: `[:300]` (augmenter),
  `[:500]` (reasoning/deep.py:289,357), `context[:max_context]`
  (grounding.py:285); ContextCore budgets by len//4 but drops whole layers,
  and inference.py:2192 collapses layers to bare strings so even that
  budget stops mattering.
- Final model input: KnowledgeProcessor "Knowledge context:\n- …" bullets —
  no attribution, no budget, no omitted marker. Model cannot cite or know
  anything was cut.

## Fix (design approved by user)

One contract, one budgeter, one renderer (PromptEngine precedent extended
from templates to context):

1. `ContextPart` frozen dataclass {text, source, layer, score, rank,
   locator, truncated} — dependency-free module, importable everywhere.
2. Producers emit parts (legacy `context` string keys stay, grandfathered).
3. Budgeter at the chat seam: real tokenizer count (len//4 fallback),
   score-ranked, drop lowest-ranked WHOLE parts, emit omitted_count.
   Nothing destroyed — full text stays in store, locators address it.
4. One renderer: parts → canonical context block with attribution;
   KnowledgeProcessor becomes its projection. char-slices delete.
5. context_info SSE + get_context_inspector carry parts (additive).

Web inspector view = follow-up lane (consciousness-bus pattern).

## Boundaries

- PromptEngine template ladder UNTOUCHED (renders context, not templates).
- domain/cognitive/ deletion = separate card (dead tree, zero-risk).
- inference.py diff stays additive (most-touched file — conflict cost).

## Evidence

- **Landed:** branch `feat/context-contract` pushed; **PR #214** open
  (https://github.com/DataWeX/sloughGPT/pull/214), commit `8d4a59792`
  (12 files, +1201/−108).
- **New tests:** 37 — `packages/core-py/tests/test_context_contract.py`
  (26: frozen/defaults, token estimate + tokenizer fallback, budget
  ordering/layer-beats-score/stability/whole-part-drop/omitted honesty,
  render format) + `test_context_contract_seams.py` (11: processor
  dual-path byte-identical legacy, RAG/math_rag/augmenter emissions,
  ContextCore parts incl. full-text preservation).
- **Regression:** core-py affected set **2098 passed**; server lane
  **1346 passed** (facade reach-in ratchet green after moving imports to
  `domain.inference`); root-lane affected **565 passed**; `tests/test_rag.py`
  9 passed; root CI-equivalent lane 1461 passed — only failures
  `tests/test_integration.py` (503/404 against the live `:8000`, which runs
  a **different worktree** `quick-island`/`fix/model-readiness-verdict`
  mid-model-load; CI skips it with no server). Environmental, pre-existing.
- **Lint/typecheck:** ruff check + format CI gates green (`packages/core-py/`,
  `apps/api/server/`); mypy clean on new/changed code (one fix: typed
  comprehension in `knowledge.py`; remaining `inference.py` errors
  pre-existing, CI mypy is `|| true`).
- **Benchmark (new):** `scripts/benchmark_context_contract.py` — contract
  path **~96µs mean / 167µs p95 @50 parts** vs ~30µs legacy; 1ms exit guard
  passes; sanity 50 parts → 10 selected / 40 omitted / 474-of-512 tok.
- **Content-neutral proof (math_rag format churn):** token-level diff vs
  main = 3 insertions (import, `"parts": parts`, comma), 0 modifications;
  `_STOP_WORDS` frozenset identical (108 elements both sides).
- **Main-side red fixed en route:** `529747c9a` (ai fail-fast) left 8 long
  `patch.object` lines in `packages/core-py/tests/test_shell_repl_more.py`
  failing `ruff format --check` — main CI/CD lint red at `9002f1a09`, PR #214
  lint failed on an untouched file. Fixed directly on main `d8d2f746d`
  (8 trailing commas, AST-identical, 4060 passed), PR rebased + re-gated.
- **Pitfalls recorded:** `.git/info/exclude` bare `models` → `git add -f`
  for tracked `domain/models/*`; lint-staged prints ignore-FAILED
  cosmetically (verify `git diff HEAD` empty); `benchmark_results.py
  record` has a closed `--kind` enum (script guard instead).
- **Kanban mechanics (finish-up, 2026-10-28):** the installed `planner`
  console script is broken (`planner.cli:main` vs package `app_planner`) —
  use `PYTHONPATH=packages/app-planner/src python -m app_planner.cli …`.
  `board add` leaves new cards UNCHAINED (`verify_chain [-1]`) until
  `store.compute_chains()` reseals — `sync()` does this, the `board add`
  CLI does NOT. Note↔card linkage = `board: <uuid>` frontmatter (title
  fallback is unreliable — many older notes/cards pair up by neither).
  Full `sync()` only touches cards whose title matches a note — safe for
  note-less cards, but never run it against a dir of stale-status notes
  (foreign-flip incidents `dcbf8020a`, 2026-10-16). This lane used
  surgical `board add` ×3 instead: cards `b46de5eb` (done),
  `b52023b2` / `58527152` (todo), all three `verify` clean.

## Landing

- **MERGED — PR #214, merge commit `19efe367a`** (2026-10-28), all 16 CI
  checks green on the final head (lint, test-core-py 29m52s, test-root,
  test-server, contract gate, frontend/E2E/Next/TS).
- Branch `feat/context-contract` (4 commits: contract, AGENT_SYNC,
  count-correction, incident doc) → main via merge commit.
- Follow-up lanes open: web context-composition view over
  `context_info.parts`; `domain/cognitive/` deletion (separate card).
