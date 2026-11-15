---
id: 20261103_071231_featcontext-wire-trait-weight-managers-into-contextcore-move
status: done
board: 20261103_071231_featcontext-wire-trait-weight-managers-into-contextcore-move
title: "feat(context): wire trait-weight managers into ContextCore, move volatile context to the prompt tail, dark-flag the sampler nudge"
tags: ["context", "inference", "kv-cache", "feedback"]
sprint: ""
gh: null
---

# feat(context): wire trait-weight managers into ContextCore, move volatile context to the prompt tail, dark-flag the sampler nudge

Two feedback->behaviour integrators exist and only the wrong one is live.

INTENDED (user ruling): feedback -> TraitWeightsConfig.update() -> managers read weights -> ContextCore applies -> prompt.
The pieces exist: domain/context/_internal/managers.py has TraitWeightsConfig (clamped [0,1] by construction, persisted
MogDB+JSON, snapshots, 5s cache) + Personality/Memory/Style/Task managers with apply(base_prompt);
domain/inference/_internal/context_contract.py owns ContextPart provenance + budget_parts + render_context_block;
ContextCore.build_context_frame is live in inference.py:614 and feedback/workflow.py:177 already calls update_from_feedback().
MISSING LINK: ContextCore.set_managers() has ONLY test callers, so _apply_managers() never runs in production and the frame
carries no trait steering.

DUPLICATE (accidental): domain/feedback/_internal/meta_weights.py integrates feedback into SAMPLER params
(_apply_meta_weights x4 in inference.py). Unbounded floats into a shared, model-agnostic gain -> rep 0.8 -> reward
repetition -> the loop in card 20261102_001621. Duplicates a responsibility TraitWeightsConfig already owns, one layer down.

SECOND DEFECT: context is injected at the HEAD (knowledge appended to system_prompt :1993, RAG messages.insert(0,...) :2020),
so the token prefix changes every turn and kills the KV prefix reuse measured at 43s -> 1.26s. Steady-state prefill is
re-paid per turn on a 482-token prompt.

Plan: (1) attach_managers(ctx_core, config) in domain/context + startup hook; (2) SLO_META_WEIGHT_NUDGE default off;
(3) stable head (system_prompt + agent instructions) vs volatile tail (knowledge/frame/RAG appended to the LAST user turn);
(4) tests: managers reflected in the frame, flag-off returns params unchanged, head bytes identical across turns with
different context, volatile block present at the tail only; (5) benchmark_latency/slonet before-after.
Supersedes 20261101_182428 (the two-bound-set question dissolves: bounded trait space is the authority; sampler nudge goes dark).
Evidence bar: same answer with feedback ON vs OFF for the count-1-to-10 prompt, and reused-token count preserved.

LANDED on main `e3fce3a01` (4 commits: 93f3b1196 feat, af67cfc2c/f019659ea-family + e3fce3a01 rehash/sync; fix commit 71ef98fa4).

WHAT SHIPPED
1. `ensure_managers(core, config)` in `domain/context/_internal/managers.py`, exported from
   `domain.context`, called from `InferenceRouter._get_context_core()` (fail-open). The seam
   `set_managers()` that only tests had ever called is now live.
2. `SLO_META_WEIGHT_NUDGE` (default "0"): `_apply_meta_weights` returns the caller's params
   untouched. Feedback authority is single-sourced in `domain/context`. Code + tests kept so
   `=1` restores the old path for A/B.
3. `_append_volatile_context()`: knowledge / context frame / RAG ride the LAST user turn;
   applied before `_trim_messages_to_budget`. No more head mutation, no `insert(0, ...)`.
4. `ContextFrame.system_extra` + frame built even when the knowledge store is empty
   (`include_rag`/`include_memory` switch off instead), and `stream_turn` appends the steering
   to the stable system head. Without this the wiring was decorative — `stream_turn` read only
   the retrieval layers and `_build_context_frame` skipped the build at `total_facts == 0`
   (the fresh-install default).

EVIDENCE
- Live probe on the real singleton: managers attached `_personality/_memory/_style/_task`;
  `build_context_frame` 1135 chars vs 140 base = +995 chars of trait steering in the prompt.
- Nudge dark -> params byte-identical; `SLO_META_WEIGHT_NUDGE=1` -> capped band
  (temp 1.0 / top_p 1.0 / top_k 28 / rep 1.0), never the old 0.8.
- `scripts/benchmark_prefix_stability.py` (new, no model/GPU): head-fold pins reuse at the
  system prompt (48 tokens, flat over 10 turns); tail-attach grows 44 -> 129 tokens = 2.7x,
  gate PASS at >= 2.0x.
- `scripts/benchmark_kv_reuse.py --stack`: unchanged (35 reused tokens / 3 turns, 100%
  consistency).
- Tests: `tests/server` 2097 passed; closure 324 passed; context core 115; domain baseline 8;
  `domain_baseline_rehash.py --check` clean; ruff check+format clean.

STILL OPEN (needs a human decision, not code)
- The live `:8000` process still runs the pre-fix tree — `systemctl --user restart slough-api`
  is required for any of this to be observable in the running app.
- Disk at 98% (2.5G free) blocks `ruff format --check .` repo-wide; package caches
  (`conda clean -a` ~819M, `~/.npm` 266M, `~/.cache/ms-playwright` 863M) were identified but
  NOT reclaimed without approval.
- Same-lane follow-up: `/v1/chat/completions` never reads the context frame at all (it is why
  it never looped). Threading trait steering through it is a separate card.
