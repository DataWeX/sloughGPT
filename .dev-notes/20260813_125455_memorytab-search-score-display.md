---
id: 20260813_125455_memorytab-search-score-display
title: MemoryTab search score display
status: done
tags: memory,chat
created: 2026-08-13T12:54:55.181243+00:00
---

MemoryTab search score display

Mirrored MemoryCard's search score display: when searchResults !== null, each result row shows its relevance score as {item.score.toFixed(2)} (text-[10px] font-mono) in the actions column. Hidden in browse mode and cleared when search clears. 2 new tests. tsc 0; MemoryTab 53/53; panels 131/131. MemoryTab now mirrors all core MemoryCard capabilities: search, metadata, show-all, click-to-copy, store, topic filter, sort, edit, score.