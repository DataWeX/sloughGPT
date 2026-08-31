---
id: 20260813_093517_memorytab-click-to-copy
title: MemoryTab click-to-copy
status: done
tags: memory,chat
created: 2026-08-13T09:35:17.063183+00:00
---

MemoryTab click-to-copy

MemoryTab facts now copy to clipboard on click/keyboard (title='Click to copy', cursor-pointer, role=button+tabIndex+Enter/Space). Transient 'Copied' indicator (1500ms, text-success) replaces global toast since chat page mounts no toast store toaster. Failure path logs via logger.debug. 4 new tests (click, keyboard, auto-clear with fake timers, clipboard-unavailable). tsc 0; MemoryTab 35/35; panels 113/113.