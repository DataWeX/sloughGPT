---
id: 20260813_092808_memorytab-item-metadata-source-badge-relative-time
title: MemoryTab item metadata (source badge + relative time)
status: done
tags: web,memory,chat,ux
created: 2026-08-13T09:28:08.447106+00:00
---

MemoryTab item metadata (source badge + relative time)

Rich-card metadata row on chat-sidebar MemoryTab items: source badge (mirrors knowledge-page MemoryCard), relative time via reused lib/format-bytes formatRelativeTime (title shows exact datetime), hidden when timestamp is 0. 2 new tests (29 total in file). tsc 0 errors; panels suite 107/107 pass.