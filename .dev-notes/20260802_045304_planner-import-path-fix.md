---
id: 20260802_045304_planner-import-path-fix
title: Planner import path fix
status: done
tags: 
created: 2026-08-02T04:53:04.850665+00:00
---

Planner import path fix

test_kanban.py and test_notes.py referenced stale sys.path tools/planner/src; planner moved to packages/planner/src. Fixed both. Full core-py collection now clean: 6996 tests collected, 0 errors (was 1 collection error).