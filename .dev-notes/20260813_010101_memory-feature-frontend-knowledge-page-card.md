---
id: 20260813_010101_memory-feature-frontend-knowledge-page-card
title: Memory feature frontend (Knowledge page card)
status: done
tags: frontend,memory,knowledge
created: 2026-08-13T01:01:01.070971+00:00
---

Memory feature frontend (Knowledge page card)

Increment 2: per-item memory deletion. Backend: added DELETE /memory/{item_id} to routers/memory.py (thin wrapper over existing MemoryService.delete). Frontend: memory-controller.delete() + MemoryDeleteResult, MemoryCard gains a per-row trash button (hover-reveal, group pattern) and a confirm AlertDialog, filtering the active search results before refetch. Tests: 4 new router tests (17 total), 2 new controller tests (13 total), 1 new MemoryCard test (14 total), knowledge page suite green. tsc clean; all memory service core tests pass (exit 0).