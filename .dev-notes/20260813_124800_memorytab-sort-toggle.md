---
id: 20260813_124800_memorytab-sort-toggle
title: MemoryTab sort toggle
status: done
tags: memory,chat
created: 2026-08-13T12:48:00.936254+00:00
---

MemoryTab sort toggle

Added newest/oldest sort toggle to the chat sidebar MemoryTab (mirrors MemoryCard): IconClock + 'Newest'/'Oldest' ghost button in the search row, aria-label 'Toggle memory sort order'. browseList useMemo sorts items by timestamp (newest default); topicFiltered now derives from browseList so sort applies to the browse list while search results stay relevance-ordered (toggle disabled during search, matching MemoryCard). 3 new tests (oldest-first ordering via listitem order, sort-state label, disabled while searching). tsc 0; MemoryTab 46/46; panels 124/124.