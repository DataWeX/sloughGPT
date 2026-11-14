---
id: d94aea1c
title: "Feature-flag roadmap: decide the 19 unread flags, then give web/mobile/gateway a flag surface"
status: review
board_id: Main
date: 2026-10-14
branch: feat/flag-surface
tags: [gene, product, flags, apps-api, domain-shared]
---

# Card d94aea1c — feature-flag roadmap (batch of 2026-10-14)

Worktree `.wt-flags`, branch `feat/flag-surface` (base c6d95a9e2).
Dependent discussion card: **fb2805c0** (flag surface — same worktree/branch).

## Scope guard

- **NO flag deletions in this batch** — deletion needs explicit user approval.
  The decide half ships a decision memo marked *NEEDS USER SIGN-OFF*.
- Surface half per the mechanism already agreed on card fb2805c0:
  sync in-memory evaluator in domain/shared; refresh fetch at the app-host
  async seam (apps/api/server/infrastructure/startup.py — never domain/
  fetching over HTTP, core stays sync); offline fallback to
  config/feature_flags.json; clients evaluate a local snapshot, never fetch
  per read. ONE descriptor-projected route (ToolSpec-shaped descriptor +
  create_router), NOT an endpoint-per-flag switchyard. Frontend goes through
  http-client.ts.
- Port-don't-rebuild: commit **89e7ab607** "fix(flags): config/feature_flags.json
  was never loaded — it was inert" (+ its branch siblings) from
  `feat/per-app-feature-flags`.

## Verified truth (from card)

Production reads exactly ONE flag: `native_c_inference`
(apps/api/server/infrastructure/startup.py, domain/models/_internal/provider/setup.py).
25 of 26 flags gate NOTHING. Tooling: scripts/check_feature_flags_usage.py (+ --strict).

## Deliverables

1. Decision memo (keep/wire/kill per flag, evidence) — appended below,
   marked NEEDS USER SIGN-OFF.
2. Flag snapshot surface (evaluator snapshot + refresh seam + one projected
   route + web client snapshot helper) — commit(s) on feat/flag-surface.

## Status log

- 2026-10-14: claimed by Gene batch 2026-10-14 (coordinator).

## DECIDE HALF — decision memo (2026-10-15, landed e855a3f4a on feat/flag-surface)

Tooling ported from **2054d15b6** (provenance-only resolution — per-app feature NOT
smuggled in; deviations recorded in the commit). `scripts/check_feature_flags_usage.py`
rc=0: **26 declared / 1 read in production / 25 unread**. Evidence axis: word-boundary
`git grep` over tracked production files (`domain/`, `apps/api/`, `packages/`; tests,
registry, config and doc excluded).

### WIRE — capability exists, flag gates nothing (14)

| flag | evidence (production files, word-boundary) |
| --- | --- |
| multimodal | 42 (middleware, startup, routers/_manifest) |
| quantization | 61 (config, health, startup) |
| slonet_provider | 17 (controllers/models, provider area) |
| knowledge_memory | 20 (routers/inference, session, prompt_builder) |
| model_registry | 13 (health, controllers/models) |
| kv_cache | 11 (inference session_kv_manager, slonet_provider) |
| model_server | 10 (health, main.py, response_cache) |
| dpo | 8 (feedback, multimodal router, training) |
| vlm | 5 (routers/multimodal, training schemas/visual) |
| semantic_cache | 3 (inference semantic_cache.py, response_cache) |
| cross_attention | 1 (multimodal/_internal/config.py) |
| hf_finetune | 1 (training/builds.py) |
| slonet_kernels | 1 (training/_internal/slonet.py) |
| soul_manager | 1 (inference slo_manager.py) |

Recommendation: populate `owner`/`remove_when` (fields landed) and gate each at its
natural read site as that area is next touched — no big-bang. Deleting any of these is
**NEEDS USER SIGN-OFF**.

### WIRE-WHEN-LANDED (1)

- `process_isolation` — capability in flight on `feat/isolated-generation` (card
  b6cf5e42, cut-over + `SLO_ISOLATED_INFERENCE` env gate). At land time either the
  registry flag becomes the gate or the flag dies and the env var stays. Decision rides
  b6cf5e42's landing.

### DELETE/MERGE candidates — no capability behind the name — NEEDS USER SIGN-OFF (10)

- **zero production evidence**: `cloud_vector_store`, `context_managers`, `llm_nlp`,
  `on_device_training` (training exists; the *on-device* variant the flag names does
  not), `soul_format` (0 hits; souls exist — likely merge into `soul_manager`),
  `feature_flags` (self-gate; delete)
- **merge into kv_cache**: `session_kv_cache` (one kv manager module, one flag suffices)
- **test-wave artifacts** (0 hits; the latter two not even in config/feature_flags.json):
  `slonet_provider_tests`, `slonet_provider_wave_i`, `slonet_wave_f`

### Notes

- `config/feature_flags.json` holds 24 flat entries; `slonet_provider_wave_i` and
  `slonet_wave_f` are not configured at all.
- The config currently asserts capabilities that do not exist for ~10 names — per the
  doc's own roadmap step 1 that is worse than no file; the delete list is the fix.
- Surface half unchanged: fb2805c0's mechanism stands (sync evaluator in domain/shared,
  refresh at the app-host async seam, offline fallback to config, ONE
  descriptor-projected route); `89e7ab607` + siblings remain stranded on
  `feat/per-app-feature-flags` and ride with that card.
- Doc roadmap item 5 (`check_feature_tags.py` counting `.wt-*` worktrees, card 3dfbbb90)
  confirmed still broken — separate card, untouched.
