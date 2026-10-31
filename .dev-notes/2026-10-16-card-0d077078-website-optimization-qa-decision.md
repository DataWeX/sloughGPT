---
id: 0d077078
title: "Website optimization via QA agent — extend the tooling or drop the expectation"
status: review
board_id: Main
date: 2026-10-16
tags: [decision, web-perf, qa-agent, product]
---

# Card 0d077078 — decision memo: website optimization via QA agent

Origin: 2026-08-28 ask assumed "website optimized via qa agent". Card 7e2af9fe
(intake check, now done) established the capability never existed. This memo
closes the fork: **extend the qa tooling (a) or drop the expectation (b)**.

## What the investigation found (re-verified today)

1. **The qa agents are core-backend by their own contract.** qa-developer's
   agent definition: "QA developer agent for core backend infrastructure …
   coverage in `domain/`". qa-verifier is its gate partner. Neither has a
   web/perf surface; "website optimization mode" has no home lane today.
2. **The expectation is documented nowhere.** grep for
   `lighthouse|vitals|perf budget|performance budget` across `docs/` → zero
   hits. It is a stale planning artifact, not a product requirement
   (PRODUCT_ENGINEERING.md / ROADMAP.md carry no such line).
3. **Nobody ever tried it.** `git log --all --grep "lighthouse|bundle-size|vitals"`
   → zero relevant commits on any branch (only unrelated next-compat shims
   for a web-vitals *module* export).
4. **No measurement substrate exists.** CI's web job runs `tsc --noEmit`
   only — there is no web build step, hence no artifact, no size baseline,
   no vitals capture. apps/web has no size-limit/visualizer tooling.
5. **Real web perf work here has been surgical and symptom-driven**, not
   agent-driven: `7365b757c` (chat: Zustand store, Virtuoso virtualization,
   lazy Prism) and `29510f6f7` (RAG documents pagination, 3.0 MB → 31 KB —
   99x). Each was triggered by a measured user-visible symptom in the lane
   that owns the journey.

## Options

| | Cost | What it buys | Risk |
|---|---|---|---|
| **(a) extend** | lighthouse-ci + budget files + trends storage + a runner lane + new agent scope; no component exists to reuse (fact 4) | automated perf signal | scope creep with no acceptance criteria (the card itself says criteria needed first); a dashboard nobody owns |
| **(b) drop** | zero | honest close of a stale expectation | none — symptom-driven perf work continues in feature lanes as it always has |
| **(b′) minimal guard** | one CI web build step + a bundle-size budget check (~one script) | catches the one real regression class: an accidentally-huge dependency in the initial chunk | small maintenance; still not an agent |

## Recommendation: **(b) drop, with (b′) as the optional guardrail**

The ask inverted how web performance actually works in this product: perf is
a property of specific journeys (chat scroll, RAG list) owned by feature
lanes, measured when a symptom appears — not a standing optimization loop.
An agent-shaped "website optimization mode" would be built against zero
reuse surface, with no user persona criterion backing it (Jon/Maya criteria
in USER_PERSONA.md name correctness and latency of the backend, not a
lighthouse score) and no docs carrying the expectation. Drop the line.

If a guardrail is wanted anyway, take **(b′) only**: add the web build to CI
and fail on a size budget. It covers the one failure mode a budget can
actually catch (bundle bloat), needs no agent, and reuses the existing
`npm ci --ignore-scripts` setup in ci.yml. (a) should be rejected unless a
concrete journey-blocking symptom recurs twice — the historical trigger
pattern for perf work here.

## Decision needed (user)

- **(b) close the line** (recommended — no further action), or
- **(b′) also add the bundle-size CI guard** (I implement it: CI build step
  + budget check + a budget file), or
- **(a) full website-optimization mode** — needs your scope + acceptance
  criteria before any implementation.
