---
id: 20260813_125847_memorytab-consolidate
title: MemoryTab consolidate
status: done
tags: memory,chat
created: 2026-08-13T12:58:47.320700+00:00
---

MemoryTab consolidate

Added one-click Consolidate to the chat sidebar MemoryTab (mirrors MemoryCard): compact text button in the header next to the fact count, shown only when memory is enabled and facts exist, title 'Merge near-duplicate facts'. Calls memoryController.consolidate(); result shows as a transient inline line below the header (3.5s auto-clear, tab's no-toast pattern): 'Consolidated N duplicate fact(s), kept M' or 'No near-duplicate facts found'; failures show 'Failed to consolidate memory'. Button disables with 'Consolidating…' while pending; refreshes list after. 5 new tests. tsc 0; MemoryTab 58/58; panels 136/136.