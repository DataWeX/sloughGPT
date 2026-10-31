---
title: 2026-10-24 — FRONTEND BRIEF: surface consciousness reflection on the webchat page
status: open
date: 2026-10-24
tags: [frontend, chat, consciousness, brief, handoff]
lane: frontend (sloughGPT-frontend worktree, feat/frontend)
---

# FRONTEND BRIEF — webchat: surface the reflection feature (user request)

**Why this note exists**: the user asked (2026-10-23) to make the frontend agent
aware of the reflection→prompt feature we built and have it on the webchat page.
The subagent dispatch failed on the free-usage limit — this note is the durable
handoff; any frontend session can work it. Card it per SOP in your worktree first.

# What we built (backend — awareness only, do NOT modify)

**Reflection→prompt injection, "CONSCIOUSNESS_PROC" (card 243016c3, commit
`3e0885625` on feat/npu-caps — stack branch, NOT yet on origin/main).**

Each chat turn, when consciousness is enabled (`config.is_enabled()` = level > 0),
the engine's **propose-only** reflection narrative (`engine.reflect(apply=False)` —
never commits belief deltas) is prepended to the model's prompt as a system message
(`Self-reflection context:\n…`) right after the KNOWLEDGE_PROC pipeline step, before
generation. Advisory by design: disabled engine, empty narrative, or ANY failure
silently skips injection — chat never errors on reflection.

Implementation (data path context):
- `ConsciousnessProcessor` — `domain/models/_internal/provider/processors/consciousness.py`.
  Takes the narrative as PLAIN TEXT (models stay decoupled from cognition).
- `_apply_consciousness_reflection(messages, corr_id)` helper in
  `apps/api/server/routers/inference.py` — `domain.core.get_consciousness` +
  `is_enabled()` guard, `apply_processors` seam, try/except skip.

# Existing webchat surface (audit before building — do NOT rebuild)

Two `CONSCIOUSNESS` SSE events per chat turn (both already on origin/main):
1. **Pre-generation**, status `"active"`, data `{level, qualia, beliefs}`
   (inference.py ~2399).
2. **Post-generation**, status `"complete"`, data `{level, self_insight}` where
   `self_insight` IS the reflection narrative, message "Reflection updated"
   (from `_run_post_gen_tasks`, ~2831).

Client plumbing that already exists:
- `apps/web/lib/stream-chat-response.ts` parses BOTH events (level, qualia,
  beliefs, growth_delta, self_insight) → `emitConsciousness()` on the bus +
  optional `onConsciousness` callback.
- `apps/web/lib/consciousness-bus.ts` + `apps/web/hooks/useConsciousnessLive.ts`.
- `apps/web/features/chat/ChatPage.tsx`: `consciousnessOpen` toggle →
  `ConsciousnessChatPanel` drawer; also `ConsciousnessMessageBadge` and
  `ConsciousnessToggle` in `apps/web/features/chat/components/`.
- **Known gap**: no production component consumes `self_insight` (the actual
  reflection text) — only tests reference it. The reflection is streamed to the
  client but never shown.

# The ask (user's words, decoded)

"We would like to have [the feature] on the webchat page." — make the reflection
visible in the chat experience: the user should be able to see, on the chat page,
what the engine reflected (the `self_insight` narrative per turn and/or the live
consciousness state). Design call is yours within the existing chat page (NO new
page — compose into ChatPage/panel/badge). Suggested direction, your call:
surface the latest reflection in `ConsciousnessChatPanel` and/or a subtle
per-turn affordance near the message it informed. The prompt-side injection
itself is invisible by design (it's for the model, not the user).

# Constraints (repo rules)

- http-client only for API calls; Noir Violet tokens only (no hardcoded hex/
  spacing); strui components where they fit; GlobalBanner/useBannerStore for
  app-wide banners (not per-page).
- Full state matrix on anything interactive (hover, focus-visible, active,
  disabled, loading, success) + ARIA.
- Tests for what you change (vitest + RTL); tsc + eslint clean on your branch.
- `feat/frontend` is mid-Vite-migration (Next removed) — the chat page may be
  in flux.
- If backend data seems missing for a UI you want, STOP and report back —
  backend is the main session's lane. (No payload change should be needed:
  level/qualia/beliefs pre-gen, narrative as self_insight post-gen.)
- Update docs/AGENT_SYNC.md in your push, per sync rules.
