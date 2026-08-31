---
id: 20260813_091454_memorycard-sort-toggle-click-to-copy
title: MemoryCard sort toggle + click-to-copy
status: done
tags: web,memorycard,ux
created: 2026-08-13T09:14:54.487980+00:00
---

MemoryCard sort toggle + click-to-copy

MemoryCard: Newest/Oldest sort toggle (default newest-first, client-side timestamp sort; disabled during search since relevance order wins) + click-to-copy on fact rows via navigator.clipboard with toast feedback. 2 new tests: sort order flip, clipboard copy. Verified: 42/42 MemoryCard tests, tsc exit 0, eslint exit 0.