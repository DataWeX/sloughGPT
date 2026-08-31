---
id: 20260813_092850_edit-memory-facts-end-to-end
title: Edit memory facts end-to-end
status: done
tags: web,memorycard,edit
created: 2026-08-13T09:28:50.464599+00:00
---

Edit memory facts end-to-end

Edit memory facts end-to-end: KnowledgeMemory.update_fact (same-id upsert, embedding recompute, dedup guard) + MemoryService.update + PATCH /memory/{item_id} ({updated, duplicate}) + memoryController.update + MemoryCard edit UI (per-row edit, Save/Cancel, duplicate toast). Core 39 tests, router 31, vitest 90, tsc/eslint clean.