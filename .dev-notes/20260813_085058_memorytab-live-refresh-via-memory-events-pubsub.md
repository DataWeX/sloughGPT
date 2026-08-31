---
id: 20260813_085058_memorytab-live-refresh-via-memory-events-pubsub
title: MemoryTab live refresh via memory-events pub/sub
status: done
tags: frontend,memory,chat
created: 2026-08-13T08:50:58.482781+00:00
---

MemoryTab live refresh via memory-events pub/sub

Created lib/memory-events.ts pub/sub. useChatMessages onMemory now publishes the event; MemoryTab subscribes and refetches when stored=true, so the panel count/list updates live without manual refresh. Tests: 5 lib tests, 2 new MemoryTab tests (17 total), all 48 targeted tests pass; tsc clean on changed files.