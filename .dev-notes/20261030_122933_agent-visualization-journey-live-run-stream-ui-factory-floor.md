---
id: 20261030_122933_agent-visualization-journey-live-run-stream-ui-factory-floor
status: done
board: 20261030_122933_agent-visualization-journey-live-run-stream-ui-factory-floor
title: "Agent visualization journey: live run-stream UI (factory floor view)"
tags: ["agents", "ux", "web", "area:web"]
sprint: ""
gh: null
---

# Agent visualization journey: live run-stream UI (factory floor view)

User vision delivered: the factory-floor view (ux/agents "Watch the Factory") — live run-stream UI composed onto /agents run history as a third view mode (list/timeline/floor). Surface ladder honored: zero new routes, page-baseline.json untouched (page-journey gate 6/6), no model names/tok-s/raw SSE frames shown.

DONE evidence (branch feat/agent-viz, commit b0568403a, 12 files +1139/-61):

Phase A — journey: docs/UX_FLOWS.md "### Watch the Factory — ux/agents" (persona Maya/Jon operator, 3-click flow, should-NOT-see list: raw frames, refresh-only updates, ghost activity/two writers; compose verdict + no page-baseline growth recorded inline).

Phase B — contract pin + doc: TestExecuteStreamVocabulary (START/TOOL_CALL/TOOL_RESULT/FINAL/COMPLETE + ERROR path) and TestOrchestrateVocabulary (PLAN→EXECUTE→COMPOSE→COMPLETE, plan-failure) in apps/api/server/tests/test_agents_router.py — 4 new tests; file green 30 passed (34 with test_agent_routines_contract.py). docs/routers.md: three previously-undocumented SSE endpoints added to the /agents table + "Run-stream event vocabulary" section (event→phase/status tables, unknown-event passthrough rule, orchestrate phase sequence, derived floor state machine with sticky terminals).

Phase C — one write authority: apps/web/app/(app)/agents/run-state.ts pure reducer (runReducer/hydrateRun/applyFrames; live-fence: hydration never clobbers live stations/phase) with red-first run-state.test.ts (8 tests, red on missing module → green); use-run-stream-store.ts zustand single writer (SSE frames → consume, GET /agents/runs → hydrate, run-switch resets per-run task-id collisions, bounded reconnect policy shouldRetry: progress buys 3 retries, no progress buys 1 — POST streams are never blindly re-fired); agents-run-floor.tsx Floor view (Noir Violet tokens, ≥11px, explicit runsLoading boolean, status as text never color alone); useAgentsWorkbench rewired: orch* state replaced by gated store selectors — the composer shows only stream-originated state (a real regression was caught by the existing "expands a run" test: hydration had started duplicating every task row in the composer; fixed with the touched-gate), Clear = one store reset, unmount resets the module store; orchestration card props reduced5 setters → clearOrchState; runViewMode union gains 'floor'. Journey tests added: floor hydrates stations from run history (incl. composer-does-not-duplicate pin) + explicit floor empty state.

Phase D — gates (all on b0568403a): vitest agents dir 56 passed (5 files incl. 2 new floor tests), page-journey-gate 6/6, agents-controller 11 passed, run-state 8 passed; pytest api 30 passed; tsc exit 0; eslint 0 errors (5 warnings, all pre-existing at lines 30/61 of untouched mocks — verified none introduced); check_surface_scope advisory only (api touch = 1 contract-pin test file, zero api shipping code — recorded rationale).

Live walk (curl, chrome-devtools unavailable): GET /agents/runs → 200, 7 runs, newest run_20261125_132031_29687_000002 completed response "done" (hydration source serving the floor); vite /agents route → 200; POST /agents/{id}/execute/stream live model: START(working) → FINAL(working) → COMPLETE(complete) ×2 — exactly the pinned vocabulary (no TOOL_* frames because no tools ran); POST /agents/orchestrate: PLAN(working) → ERROR(error) — pinned error path walked live, but the happy path is blocked by a PRE-EXISTING backend defect (every orchestrate died at plan since ≤2026-11-20: run_20261120_114456 has the same error) — MultiAgentOrchestrator._cmds = ShellCommands but ShellCommands defines neither generate nor generate_async (multi.py:146/420/429; generate_async exists only on Soul, soul.py:850). Carded separately: 20261125_122006_orchestrator-llm-seam-dead-multiagentorchestrator-calls-shel. The floor's failed-run journey step ("settles into done or failed with a visible outcome") is what the live walk proved.

Frontend-design skill compliance: Noir Violet tokens only, type floor ≥11px, loading gated by explicit boolean (never content emptiness), status rendered as text + dot (not color alone), no raw event vocabulary shown to users.
