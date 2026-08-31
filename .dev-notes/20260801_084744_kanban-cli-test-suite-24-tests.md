---
id: 20260801_084744_kanban-cli-test-suite-24-tests
title: kanban CLI test suite — 24 tests
status: done
tags: planner,tests,kanban
created: 2026-08-01T08:47:44.842014+00:00
---

kanban CLI test suite — 24 tests

Added packages/planner/tests/test_kanban.py (31 tests): CLI-driven init/force, add defaults+fields, invalid priority (SystemExit 2), list filters+empty, show/edit/move/delete + unknown handling, board render, note add/list/delete + unknown card, column management (add/rename/rm/duplicate), archive, search, stats, no-command; plus store-level edge cases (invalid priority fallback, ambiguous prefix None, update_card unknown keys, column migration on rename/remove, empty archive, assignee filter). README dev section updated. Full planner suite: 127 passed.