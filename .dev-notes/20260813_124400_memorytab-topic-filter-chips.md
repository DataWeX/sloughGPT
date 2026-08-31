---
id: 20260813_124400_memorytab-topic-filter-chips
title: MemoryTab topic filter chips
status: done
tags: memory,chat
created: 2026-08-13T12:44:00.634416+00:00
---

MemoryTab topic filter chips

Added topic filter chips to the chat sidebar MemoryTab (mirrors MemoryCard): horizontal scrollable 'All' + topic chip row (aria-label 'Filter by topic'), derived from items via useMemo. topicFiltered combines search results + active topic; Show-all cap and count now apply per topic. Empty state: 'No memory in the X topic.' when a filtered topic has no items. 4 new tests (chip filter + All restore, chip re-click toggle, topic empty state via search exclusion, per-topic 8 cap + Show all 12). Updated pre-existing assertion (getByText->getAllByText for 'preferences') since chips and item badges now share topic text. tsc 0; MemoryTab 43/43; panels 121/121.