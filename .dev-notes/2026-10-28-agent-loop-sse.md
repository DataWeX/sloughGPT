---
title: "Agent loop over the live provider: Qwen adapter + SSE event route"
status: done
board: 3fa21636-9fce-4fde-8876-8afadf1bc4f7
date: 2026-10-28
tags: [backend, agents, inference, sse, tool-calling]


# AgentLoop × live provider × SSE

## What

Wire the canonical `AgentLoop` (already merged, tool-spec driven, event
emitting) to the live registered model provider (autoloaded Qwen 2.5 /
SmolLM / SloNet behind `get_provider`), and stream its events over SSE so
the web can graph executed tools.

1. `domain/agents/_internal/provider_infer.py` — `make_provider_infer()`
   wraps any `ModelProvider` into the loop's `InferFn` seam:
   - teaches the canonical ```tool_call protocol via system-prompt
     injection (same approach as `ToolUseProcessor` for the legacy marker);
   - translates `tool`-role messages to attributed `user` messages —
     `SloTransformer._messages_to_prompt` silently DROPS unknown roles, so
     without this the model never sees its own tool results.
2. `ToolRegistry.subset(names)` — filtered registry for the route;
   unknown names raise, never degrade to all-tools.
3. `POST /agents/loop` (SSE) — streams `plan → tool_call/tool_result* →
   final` as `agent-loop` frames; no-provider → in-band 503 frame; unknown
   tool → 400 frame; `requires_approval` tools fail honestly (no mid-stream
   approver; approve=None is the loop's explicit denial).
4. Facade fix: `domain/agents/__init__.py` had TWO `__getattr__`s — the
   second (merge artifact) shadowed the first, so the documented
   `domain.agents.AgentLoop` was an AttributeError. Merged into one.

## Why

- Follow-up to the context-contract lane: structure landed, this is the
  execution seam the user asked for ("import qwen 2.5 … work on tool
  calling … graphing of executed tools" in the web).
- The loop was built model-agnostic (caller-supplied `infer`); this fills
  that seam with the real provider instead of test fakes.

## Landing

- **PR #223 MERGED `c05af116c`** (2026-10-28) — all 16 CI checks green on
  final head; branch `feat/agent-loop-sse` deleted (remote + local).
- AGENT_SYNC entry `75c04362b` (direct push, tip verified unchanged in
  same breath).
- Tests: 23 new adapter tests (`packages/core-py/tests/test_provider_infer.py`
  — contract rendering, tool-role translation, resolution order, kwargs
  forwarding, **full tool round-trip with a Qwen-shaped fake**, subset,
  facade exports) + 3 SSE route tests in `test_agents_router.py`; 629
  agent-suite tests green; mypy clean on new/changed code; ruff clean.
- No benchmark: wiring seam, not a perf-sensitive path (one string build +
  list copy per infer call vs model-generation cost).

## Pitfalls

- **test-web flake signature (again):** first run red with 808/808 files +
  7732/7732 tests passed — 2 unhandled rejections from
  `settings/page.test.tsx` teardown → `error-store.ts:91
  getSeverity(undefined)` (identical to the stale-PR-batch entry's triage).
  Python-only PR, zero causal path → `gh run rerun 40628901408 --failed`
  after the run **completed** → green. The rerun only works once the run
  is `completed`, not while in progress.
- **pre-commit reformats committed markdown:** husky → `npx lint-staged` →
  `prettier --write` on `*.{json,md}` canonicalizes emphasis
  `*italics*` → `_italics_` (remark-stringify; `**strong**` untouched,
  proseWrap preserve = no reflow). My AGENT_SYNC commit silently changed 3
  neighbor entries' emphasis markers — content-neutral, but review
  `git show` of docs commits in raced files.
- **Facade duplication is invisible to grep:** two `__getattr__`s in one
  module file — the second legally shadows the first; nothing errors until
  a consumer hits AttributeError. Check for duplicate top-level defs when
  a documented facade export mysteriously fails.

## Follow-up (not in this PR)

- Web consumption of the `agent-loop` SSE stream (frontend lane —
  consciousness-bus pattern, compose-first per Page Rule).
- Native Qwen function-call chat templates deliberately NOT used — the
  adapter teaches the protocol in-prompt (zero template coupling, works
  with any chat model).
