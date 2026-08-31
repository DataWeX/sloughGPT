---
id: 20260813_090038_topic-filter-on-memory-list
title: Topic filter on memory list
status: done
tags: memory,ui,knowledge
created: 2026-08-13T09:00:38.018191+00:00
---

Topic filter on memory list

Topic filter on the Knowledge memory list (frontend-only, derived from loaded items — no new endpoint).

MemoryCard: activeTopic state + useMemo topics list (dedup + case-insensitive sort, capped at 8 chips with '+N more' overflow). Chip row below the search bar: 'All' + one chip per distinct item.topic; active chip highlights bg-primary/15 text-primary; clicking a chip toggles it, clicking again or 'All' clears. filter applies to both the flat list and live search results (filteredByTopic = (searchResults ?? items) filtered by activeTopic). Empty state is topic-aware: 'No memory in the <topic> topic.' takes precedence over the search-empty message.

Tests: +4 (chips derived from items incl. second topic fixture, filter narrows the list, topic-aware empty message when search+filter yields none, All resets). 37/37 MemoryCard, 47/47 knowledge suite, tsc + eslint clean.