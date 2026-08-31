---
id: 20260813_090208_memory-feature-live-fact-highlight-in-memorytab
title: Memory feature: live fact highlight in MemoryTab
status: done
tags: web,memory,feature
created: 2026-08-13T09:02:08.206108+00:00
---

Memory feature: live fact highlight in MemoryTab

Increment: MemoryTab now highlights the newly stored fact when a MEMORY event arrives.

- MemoryTab.tsx: pendingFactRef captures the fact from the memory event; fetchData consumes it after refetch, matches item by exact content, sets highlightedId, auto-clears after 4s (HIGHLIGHT_MS) via highlightTimerRef with unmount cleanup. List li gets border-primary/60 + bg-primary/10 when item.id === highlightedId.
- MemoryTab.test.tsx: +3 tests (highlight matching fact, no highlight on no-match, auto-clear via fake timers). 20/20 pass.
- npx tsc --noEmit -> 0 errors.