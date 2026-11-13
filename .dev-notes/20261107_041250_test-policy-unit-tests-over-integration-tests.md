---
id: 20261107_041250_test-policy-unit-tests-over-integration-tests
status: done
board: 20261107_041250_test-policy-unit-tests-over-integration-tests
title: "Test policy: unit tests over integration tests"
tags: [testing, policy]
sprint: ""
gh: null
---

# Test policy: unit tests over integration tests

Session 2026-11-07 (user directive). New tests default to **unit tests** — pure,
isolated, deterministic — not integration tests: no engine/loop/network/process
wiring unless the wiring itself is the subject under test. Integration-style
coverage is reserved for seam/contract/gate tests (AGENTS.md Test Rule).

Rationale: the stage-4 full-record run leaned on integration-heavy suites
(training-journeys, e2e trigger, thread-teardown) that are slow and flake under
load; unit tests pin the same logic deterministically and keep the change-scoped
local loop (Test Rule clause 3) cheap.

Scope: repo-wide guidance for NEW tests, applied at intake when test briefs are
written (Gen / test-writer). Existing integration tests are grandfathered —
rewrite as unit tests only when that suite is next touched (same clause as the
router/command grandfathers). Complements, does not replace, AGENTS.md Test Rule
(owned-code, change-scoped, no redundancy).
