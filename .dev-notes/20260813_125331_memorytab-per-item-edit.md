---
id: 20260813_125331_memorytab-per-item-edit
title: MemoryTab per-item edit
status: done
tags: memory,chat
created: 2026-08-13T12:53:31.703574+00:00
---

MemoryTab per-item edit

Added inline per-item edit to the chat sidebar MemoryTab (mirrors MemoryCard). Edit button (IconEdit, aria-label 'Edit memory item') appears on hover next to delete in each list row; opening it renders an inline form (textarea 'Edit memory fact text', topic input, importance range slider 0-1 step 0.1 with live value) that pre-fills current values. Save calls memoryController.update(id, content, topic, importance); on updated>0 it patches items + searchResults optimistically, closes the form, and refetches; duplicate -> 'That fact already exists in memory'; other no-update -> 'Memory item not found'; throw -> 'Failed to update memory item'. Store and edit forms close each other. 5 new tests (prefill, edit+refresh, duplicate inline error, cancel, failed update). tsc 0; MemoryTab 51/51; panels 129/129.