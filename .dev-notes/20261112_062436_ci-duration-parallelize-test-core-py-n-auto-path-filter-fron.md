---
id: 20261112_062436_ci-duration-parallelize-test-core-py-n-auto-path-filter-fron
status: done
board: 20261112_062436_ci-duration-parallelize-test-core-py-n-auto-path-filter-fron
title: "CI duration: parallelize test-core-py (-n auto) + path-filter frontend E2E/test-web"
tags: ["ci", "testing"]
sprint: ""
gh: null
---

# CI duration: parallelize test-core-py (-n auto) + path-filter frontend E2E/test-web

Data (run 43024943888, 2026-11-11): test-core-py = 50m19s (the whale; every other test job <= 6m46s); CI wall ~30m, E2E-bound (Cypress 22 specs, 45m timeout). test-core-py installs pytest-xdist but invokes pytest WITHOUT -n.

Levers (user-approved 'both', 2026-11-12):
1. test-core-py: add -n auto (pytest-cov integrates with xdist natively). Acceptance: local full tests/core-py run (CI's exact 9 ignores) green under xdist; next full CI/CD run test-core-py <= ~15m with --cov-fail-under=50 still enforced.
2. Path-filter frontend-heavy jobs: CI/frontend-e2e and CI/CD/test-web only run when frontend-relevant paths change (apps/web/**, packages/**, package-lock.json, vite/cypress config); default to RUN when event/paths unknown (conservative — never skip a needed run).

Constraints: workflow edits, benchmark not owed (no training/inference change) — evidence = gh run view timings; TDD analogue = local xdist validation run + YAML validation; must not weaken coverage floor or drop E2E coverage on frontend changes. Sequencing: push AFTER card 20261101_120913 CI lands + closes (avoid red-on-main confusion during its closure).
