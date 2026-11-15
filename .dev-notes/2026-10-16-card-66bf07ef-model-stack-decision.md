---
id: 66bf07ef
title: "DECISION: ModelStack is built but never read — inference_context() has zero callers"
status: review
board_id: Main
date: 2026-10-16
tags: [decision, model-stack, personas, domain-infrastructure]
---

# Card 66bf07ef — ModelStack: wire it or cut it (decision memo)

## Evidence (verified 2026-10-16, branch feat/tokenizer-checkpoint-binding)

| Claim | Verification |
|---|---|
| `inference_context()` has zero production callers | `git grep ModelStack\|model_stack` outside its own module/router = only `routers/_manifest.py` (mount table) + `tests/contract/test_router_boundary.py` (name allowlist). Nothing else. |
| No UI/script/CLI client | `git grep model-stack` in `apps/web`, `apps/mobile`, `packages`, `scripts/` = empty. |
| Stack state is a fantasy | Router `_get_stack()` = per-process in-memory singleton, default base = **dummy `~/.cache/sloughgpt/models/base.soul`** (need not exist). State dies on restart. |
| Persistence unused | `ModelStack.save/load` have zero callers (only the load-failure logger matches). |
| The strict-xfail demands a reader | `feat/persona-journey-tests` → `apps/api/server/tests/test_persona_journeys.py:302` `test_something_actually_reads_the_stack`, `xfail(strict=True)`: `_find_stack_consumers()` must be non-empty — "replace the marker with a real assertion that the training entry point reads the context". |
| Persona promise is NARROW | USER_PERSONA.md Jon:55-58 / Maya:87-90 success lines = **base resolution + file immutability** ("stacked 3 caches, train still sees model"; "model file untouched"). Their failure lines are corruption/loss modes ("adding a dataset deleted my model file"). None of the success lines require knowledge-injection or dataset-feeding semantics. |

## The real gap, stated precisely

ModelStack is a **metadata plane with no data plane** — an inventory of intent
(the exact disease the Endpoint & Transport Rule names). It was built in the
router-facade era (notes 2026-09-24/26) as a facade over an imagined future.

The card frames two options. The evidence refines them:

### Option A — wire it (product call the card names: what do layers DO?)
### Option B — cut the surface (router + module + manifest entry + contract
allowlist entry; also touches the persona lane's xfail story — cross-lane)

### Option C (recommended) — phased wire: resolution seam now, layer semantics as their own decisions

**Phase 1 (small, satisfies the strict-xfail AND both persona success lines):**
1. Make the stack *real*: file-backed persistence (the already-written
   save/load, called from the router on every mutation), and default base
   resolved from the existing model registry / explicit `set_base` — **kill the
   dummy `base.soul`** (no consumer may ever resolve a model that does not
   exist).
2. One consumer: the training entry point resolves its model **through the
   stack manifest when a stack file exists**, config/CLI otherwise. Precedence
   rule (the product call, stated conservatively): *a stack file existing =
   explicit user intent and wins; absence = today's behavior unchanged.* No
   new second source of truth: config stays the default, the stack is an
   opt-in override, never a parallel truth.
3. Immutability already holds (`ModelBase.ensure_readonly`, all ops are
   registry-only) — the persona lane's real-file tests (06713695 slice 1) pin it.

**Phase 2 (separate decisions, one card each — NOT implied by Phase 1):**
- `dataset` layers → trainer: only as a *dataset adapter* into the existing
  `SloughGPTTrainer` (AGENTS: no new training loop). Needs the call: do stack
  datasets *add to* or *replace* config datasets?
- `knowledge` layers → nothing in the repo defines what a knowledge store IS
  (documents? soul? adapters?). No subsystem exists to inject into. Design
  first.
- `cache` layers → no kv-cache store subsystem is wired today (`kv_cache` /
  `session_kv_cache` are on the unread-flags list). Mapping a layer kind onto
  an absent subsystem would be another inventory of intent. Defer.

**Why not a half-wire:** a dataset/knowledge layer that the core claims but
ignores is precisely the inventory-of-intent pattern this repo is retiring.
Either a layer kind is fully consumed or it is documented as
metadata-not-yet-consumed.

**Why not B as default:** deleting a user-visible surface (even a dead one)
needs sign-off per file-safety, and the persona lane's executable criteria
*assert the stack's promise* — cutting means rewriting the personas' success
story, which is a product statement, not a cleanup.

## The call needed from the user

1. **A, B, or C?** (C = recommended: Phase 1 wire + Phase 2 as separate cards)
2. If C: approve the precedence rule (stack file wins when present, config
   otherwise).
3. Phase 2 layer-semantics questions get their own cards after Phase 1 lands.

## Status

Memo delivered → review. No code written (decision card by construction).
