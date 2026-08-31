---
id: 20260813_094022_memory-fact-importance-editing-end-to-end
title: Memory fact importance editing (end-to-end)
status: done
tags: memory,api,frontend
created: 2026-08-13T09:40:22.817014+00:00
---

Memory fact importance editing (end-to-end)

Added importance editing for memory facts across the full stack. Core: update_fact gains importance param (clamped to [0,1], None keeps existing); re-embed stays unconditional (QueryResult has no vector). Provider protocol + KnowledgeMemoryProvider + MemoryService thread importance through. Router: UpdateRequest.importance Field(ge=0, le=1) validated at boundary; PATCH passes it. Controller: update(id, content, topic?, importance?) sends importance only when finite. UI: MemoryCard edit panel adds 0-1 range slider + row badge (importance 0.X); startEdit seeds slider from item. Tests: +4 core, +3 router (incl. 422 out-of-range), +2 controller, +1 component. Gates: core 43 passed, router 34 passed, vitest 77 passed, tsc 0 errors, eslint clean.