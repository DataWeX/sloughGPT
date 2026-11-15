---
id: 20261105_100132_trim-audit-parked-user-ruling-2026-11-05-park-dont-kill-test
status: todo
board: 20261105_100132_trim-audit-parked-user-ruling-2026-11-05-park-dont-kill-test
title: "TRIM AUDIT (PARKED — user ruling 2026-11-05: park, don't kill): Test Rule clause-5 redundancy/unowned-code audit — start from docs/TEST_FEATURE_MAP.md"
tags: ["area:tooling"]
sprint: ""
gh: null
---

# TRIM AUDIT (PARKED — user ruling 2026-11-05: park, don't kill): Test Rule clause-5 redundancy/unowned-code audit — start from docs/TEST_FEATURE_MAP.md

Backlog, unowned, no timeline — user ruling 2026-11-5: PARKED, not cancelled. This is the follow-up card AGENT_SYNC promised at card 299d8d38 (change-scoped testing lane).
BRIEF: start from docs/TEST_FEATURE_MAP.md (pushed 5e01c1b3f) — it already answers: (1) concentration lever = tests/core-py/test_shell_repl_more.py 4,081 tests = 8.6% of suite, top-5 files = 14.7%; (2) 89/103 skips concentrated in 2 shell files (TTY-gated — audit 'runs where?'); (3) 17 __corepy twin pairs, name-disjoint (28 same-name tests) — each owes an assertion-level diff (same input+assertion = the real redundancy class) before any trim; (4) disabled flags have NO dead feature tests (cleared, don't re-check); (5) dpo experimental = 38 files — product-scope question, not test debt; (6) FEATURE: deletion guard covers 6 files — decide grow-or-retire.
RULES: trimming is a card per suite behind green gates (AGENTS.md Test Rule clause 5) — no bulk deletion, no weakened assertions; redundancy verdicts need assertion-level evidence, not name collisions (4,799 cross-file same-name tests are leads only).
