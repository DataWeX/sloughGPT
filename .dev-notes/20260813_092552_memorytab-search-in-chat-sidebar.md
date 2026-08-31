---
id: 20260813_092552_memorytab-search-in-chat-sidebar
title: MemoryTab search in chat sidebar
status: done
tags: web,memory,chat,feature
created: 2026-08-13T09:25:52.408681+00:00
---

MemoryTab search in chat sidebar

Added debounced semantic search to the chat-sidebar MemoryTab: compact search box (hidden when memory off), server-side /memory/search, search results replace the 8-item list, no-match state with Clear search recovery, delete filters search results, active search re-runs on memory-events refresh. 6 new tests (27 total in file). tsc 0 errors; panels suite 105/105 pass.