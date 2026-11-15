---
title: "Context inspector (web): visualize budgeted context parts over SSE"
status: todo
board: b52023b2-8b63-428c-b165-783d80deb1ea
date: 2026-10-28
tags: [frontend, inference, context, sse]


# Context inspector — web half of the context contract

Follow-up lane to the MERGED backend contract (PR #214, `19efe367a`,
2026-10-28).

## What

Surface what the model actually received: the chat pipeline already emits
`context_info` on the `chat`/`STREAMING` SSE frame with
`{layers, total_tokens, max_tokens, parts:[{source, layer, score, rank,
locator}]}` (additive since PR #214). Build the view that renders it —
which sources were selected, their ranking/locators, and what the budget
omitted (backend tracks `omitted_count` at `CONTEXT_BUDGET` debug log).

## Why

- Structure-before-viz was the agreed split: "built well so we can
  visualize it in the web UI" (backend PR 1, web = PR 2).
- The contract's whole payoff: truncation is no longer blind — parts carry
  locator + omitted_count so the UI can show exactly what was selected vs
  cut.

## Shape (per existing precedent)

- Follow the **consciousness-bus pattern** (`consciousness-bus.ts`): a
  small store/consumer over the existing chat SSE stream — no new
  endpoint, no raw `fetch` (http-client/transport rule; the data is
  already on the wire).
- Compose into an existing chat/context surface — **Page Rule**: a new
  `page.tsx` needs a named journey citation from `UX_FLOWS.md`; default is
  compose-first (panel/section of an existing page), summarize, no raw
  dumps (SOP "Summarize, don't dump").
- Data is provenance-only (no raw text payloads in `context_info.parts`) —
  keep it that way; full text stays server-side (locator-addressable).

## Boundaries

- Backend contract untouched (`domain/inference/_internal/context_contract.py`
  is merged and stable; renderer/budgeter live server-side).
- `http-client.ts` is the single source of truth for any API call — but
  prefer the SSE frame already carrying `context_info`.

## Evidence

(to fill)
