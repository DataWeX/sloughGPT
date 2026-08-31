---
id: 20260813_090841_memorycard-ux-polish-relative-time-show-all-touch-delete-cle
title: MemoryCard UX polish (relative time, show-all, touch delete, clear-search)
status: done
tags: web,memorycard,ux
created: 2026-08-13T09:08:41.317682+00:00
---

MemoryCard UX polish (relative time, show-all, touch delete, clear-search)

MemoryCard: formatRelativeTime helper (Just now/Nm/Nh/Nd ago, date fallback) replaces bare toLocaleDateString with full-date title; show-all toggle with 'Showing N of M' footer for capped 10-item list; delete button now touch-visible (opacity-60 lg:opacity-0 lg:group-hover:opacity-100); Clear search recovery CTA in search-empty state. 3 new tests (show-all expand/collapse, clear-search reset, relative timestamp). Verified: 56 tests (2 files) pass, tsc exit 0, eslint exit 0.