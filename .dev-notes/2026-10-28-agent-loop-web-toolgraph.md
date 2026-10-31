---
title: "Web tool-graph: consume the agent-loop SSE stream (executed tools view)"
status: todo
board: 6d82b764-d343-4604-b4c9-e09673afbc2a
date: 2026-10-28
tags: [frontend, agents, sse, tool-calling]


# Web tool-graph over the `agent-loop` SSE stream

Follow-up lane to the MERGED adapter + SSE route (PR #223, `c05af116c`,
2026-10-28).

## What

`POST /agents/loop` already streams the full execution graph — `plan` →
`tool_call`/`tool_result` pairs → `final` (or `error`) as `agent-loop`
frames carrying `{phase, status, message, data:{name, args, output,
success, iteration, text, truncated}}`. Build the view that renders it as
a graph/timeline: which tools ran, with what args, what they returned,
across iterations.

## Why

- The payoff half of the user's ask ("graphing of executed tools" in the
  web) — the backend emits everything; nothing consumes it yet.
- Same structure-before-viz split as the context-contract lane.

## Shape (per existing precedent)

- **Consciousness-bus pattern** (`consciousness-bus.ts`): a small
  store/consumer over `streamSSE`/`createSSEStream` — no new endpoint, no
  raw `fetch` (server-transport contract gate forbids raw egress; the
  data is already on the wire).
- **Page Rule**: compose-first — a panel/section of an existing
  agent/chat surface, NOT a new `page.tsx` (needs a named `UX_FLOWS.md`
  journey citation + route budget). Tool-run timeline inside the surface
  that launched the loop is the default.
- Summarize, don't dump (SOP): show tool name, success/fail, one-line
  result preview; raw output one click away, collapsed.

## Boundaries

- Backend route + adapter untouched (merged and stable).
- Approval UX deliberately absent server-side (`requires_approval` fails
  honestly with approve=None) — if the web lane wants an approval flow,
  that's a backend contract change → its own card, not this one.
- Related but distinct from `b52023b2` (context inspector — budgeted
  context parts on the chat frame); this one is the execution graph on
  the `agent-loop` stream. Two streams, two views.

## Evidence

(to fill)
