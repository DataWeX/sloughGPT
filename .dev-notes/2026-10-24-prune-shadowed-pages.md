---
title: 2026-10-24 — prune 36 shadowed pages (page.tsx 109 -> 73)
status: done
date: 2026-10-24
tags: [kanban, pages, cleanup, page-rule]
---

# Prune the 36 knownShadowed pages

The 36 routes recorded as `knownShadowed` in `apps/web/lib/route-contract.json`
(PR #188) are each both a live `page.tsx` AND a redirect source; `applyRedirect`
fires before routing, so the 307 makes them unrenderable — dead files that keep
the page-count ratchet inflated (109 vs the ~20 target). Cross-validated two
ways by the route-contract lane (test + independent Python derivation).

Scope: delete each shadowed route's `page.tsx` (+ colocated `page.test.tsx`),
shrink `page-baseline.json` grandfathered 109 -> 73, empty `knownShadowed`,
drop `GRANDFATHERED_CEILING` to 73, update the PRODUCT_ENGINEERING.md count.
Redirect table unchanged (URLs keep 307ing). Unblocked by #188's merge.

Gates: route-contract + page-journey-gate + redirect tests, tsc, eslint,
full web suite; then PR + CI.
