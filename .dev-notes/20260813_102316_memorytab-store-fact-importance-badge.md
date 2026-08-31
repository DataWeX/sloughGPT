---
id: 20260813_102316_memorytab-store-fact-importance-badge
title: MemoryTab store-fact + importance badge
status: done
tags: memory,chat
created: 2026-08-13T10:23:16.212220+00:00
---

MemoryTab store-fact + importance badge

Added manual Store fact to the chat sidebar MemoryTab (mirrors MemoryCard handleAdd): '+ Store' toggle in the search row, inline form (content textarea + optional topic + Save, disabled until content). On success the stored fact is highlighted via the existing pendingFactRef highlight path (matches memory-event behavior). Inline error text for duplicate/memory-off ('Already remembered (or memory is disabled)') and failure ('Failed to store fact') since the chat page mounts no toast store. Added importance badge (importance X.X) to item metadata row, matching MemoryCard. 4 new tests (store+highlight, duplicate inline error, Save disabled state, importance badge). tsc 0; MemoryTab 39/39; panels 117/117.