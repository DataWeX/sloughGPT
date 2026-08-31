---
id: 20260813_051027_memorytab-master-remember-switch
title: MemoryTab master Remember switch
status: done
tags: frontend,memory,chat
created: 2026-08-13T05:10:27.152339+00:00
---

MemoryTab master Remember switch

Added the auto-memory master switch inline to the chat tool panel MemoryTab, closing the gap where the off-state told users to navigate to the Knowledge page to re-enable it.

- apps/web/features/chat/components/panels/MemoryTab.tsx: Remember Switch in the header row (matches MemoryCard pattern, size=sm) calling memoryController.setEnabled(); off-state hint now references the inline switch; docstring updated
- apps/web/features/chat/components/panels/MemoryTab.test.tsx: strui mock gains Switch (role=switch), controller mock gains setEnabled; 3 new tests (switch reflects enabled state, toggle off calls setEnabled(false), toggle on from disabled state calls setEnabled(true))

Verification: MemoryTab 15/15 pass; features/chat/components/panels 93/93 pass. tsc errors unchanged (pre-existing, unrelated files only).