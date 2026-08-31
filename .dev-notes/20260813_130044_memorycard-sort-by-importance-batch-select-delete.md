---
id: 20260813_130044_memorycard-sort-by-importance-batch-select-delete
title: MemoryCard sort-by-importance + batch select & delete
status: done
tags: web,memorycard,ux,feature
created: 2026-08-13T13:00:44.868614+00:00
---

MemoryCard sort-by-importance + batch select & delete

Continued MemoryCard feature build.
1) Sort dropdown: replaced Newest/Oldest toggle with a DropdownMenu (trigger keeps aria-label 'Toggle memory sort order', IconClock + label + IconChevronDown). sortOrder widened to 'newest'|'oldest'|'importance'; browseList comparator adds (b.importance ?? 0) - (a.importance ?? 0) desc, missing importance sorts last. Disabled-when-searching + title retained.
2) Batch select & delete: per-row checkboxes (aria-label 'Select memory fact <content>'), selected rows get bg-primary/[0.06] border-primary/30; 'Select all (N)' label above list; action bar with Export (N) / Delete (N) / Cancel when selection active. handleBatchDelete loops memoryController.delete per id (survives per-item failures), toasts count, clears selection + searchResults, refetches. handleExportSelected mirrors handleExportMemory shape (content/topic/source). Batch delete uses a dedicated AlertDialog.
TDZ fix: toggleSelectAll moved below displayedItems declaration. Fixed knowledge/page.test.tsx strui mock to add DropdownMenu family (KnowledgePage renders MemoryCard). Full suite green: 326 files / 3854 tests. tsc + eslint clean.