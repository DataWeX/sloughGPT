---
name: stop-slop
description: >
  Use when asked to build an engine, capability, mechanism, renderer, or "the
  actual thing" — stops the reflexive answer of an HTTP endpoint plus a web page.
  Covers the boundary test (function call vs transport crossing), the good/reflexive
  worked pair, the pre-write check, and per-role scope guards. Complements
  cleanup-agent (code-quality slop) and design-system-enforcement (visual slop):
  this one is build-target slop.
---

# Stop Slop

## Overview

In this repo the path of least resistance is always the same two steps: expose an
endpoint, then build a page that fetches it. When someone asks for an _engine_ —
something that renders, computes, decides, or powers other things — that reflex
produces a **thing called an API** instead of the actual thing.

Slop here is not bad code. It is the **wrong build target**, reached without
thinking. It passes lint, it passes tests, and it is still the wrong artifact.

Three real instances from one conversation, all the same defect:

> "I want engines that render into pages and power them."
> → answered with `GET /consciousness/render` plus a page.

> "We're always building APIs instead of actual things."

> "And then web UIs are always what we build."

## The boundary test — run it BEFORE writing code

From `AGENTS.md` ("Endpoint & Transport Rule"): endpoints are projections of
contracts, not hand-built conveyor belts. Naming the boundary is part of the work.

- **Same process** → function call. No endpoint. Ever.
- **Different process** → transport crossing. An endpoint is justified, and it is
  a _projection_ of a declared contract — never an entry in a hand-written list.

The question is not "would an endpoint be convenient". It is "does the caller live
in another process". Convenience is not a boundary.

## The worked pair

Real example, this repo, consciousness lane. The ask: _"engines that render into
pages and power them."_

**The reflex (slop):**

```text
GET /consciousness/render          <- an endpoint nobody was waiting on
  -> page fetches it, derives from it, owns state again
```

Three defects in one: a transport crossing nothing, a page that re-owns the state
the engine was supposed to own, and a render step that can only be exercised
through a browser. This is the exact pattern the Endpoint & Transport Rule retires.

**The build:**

```python
# domain/cognition/_internal/consciousness/render.py
def render_mind(records, status, note) -> MindViewModel:
    """Pure function. No I/O, no transport."""
```

Callable today by a page, a test, a CLI, a script, a story — anything that needs
it. It powers whatever shows it. A route, if one is ever wanted, is a two-line
projection at the very edge and is built only when a real consumer in another
process shows up.

## The pre-write check

Answer in one line each, before the first file:

1. **What calls this?**
2. **What does this power?**
3. **Does the caller live in another process?**

If (3) is "no", there is no endpoint. Write the function.

## The second-order trap: the page that owns state

Even when the engine exists, the drift resumes at the page. A page that fetches,
derives, and stores is a private little engine, and every page becomes one. The
rule that stops it:

> The engine returns a **view-model**. The page only **paints** it.

The view-model carries both planes — the human-readable rendering _and_ the
provenance in its bones. Readability must never cost provenance, and provenance
must never leak into human prose as numbers. Two planes, one record, neither view
a lie about the other.

The failure mode to design against is an engine that "renders" by dumping raw JSON
and calling the page done. A raw dump is not a rendering.

## Per-role scope guards

Each engine-facing agent carries its own "What this role does NOT build" table
directly under `## Mission` in `.opencode/agents/`. Read yours. Common to all three:

| Reflex                                 | Instead                                                                                     |
| -------------------------------------- | ------------------------------------------------------------------------------------------- |
| An HTTP endpoint for its own sake      | A pure function the caller imports                                                          |
| A web page that fetches and owns state | A view-model the page only paints                                                           |
| A second copy of an existing engine    | Extend the shared one — check `packages/README.md`, `apps/README.md`, `docs/INDEX.md` first |

## Why the guard has to be explicit

A `## Mission` says what a role _is_; a `## Scope` table says where files live.
Neither stops the drift, because the drift is not ignorance — it is the default.
Only a stated counter-example changes the default, which is why every guard is
written as a contrast: the build next to the reflex.

## What this skill does NOT stop

- **Code-quality slop** (f-string logs, bare excepts, unused imports) → `cleanup-agent`.
- **Visual slop** (hardcoded colors, AI-default layouts, broken type scale) → `design-system-enforcement`.
- **Legitimate endpoints** — this skill does not forbid transport. It forbids
  transport chosen by reflex. When the boundary test says "different process",
  build the projection properly: one `ToolSpec`-shaped contract, emitted, not listed.
