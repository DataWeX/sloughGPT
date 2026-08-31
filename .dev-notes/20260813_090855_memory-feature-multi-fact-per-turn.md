---
id: 20260813_090855_memory-feature-multi-fact-per-turn
title: Memory feature: multi-fact per turn
status: done
tags: web,memory,backend,feature
created: 2026-08-13T09:08:55.767891+00:00
---

Memory feature: multi-fact per turn

Increment: a chat turn can store several facts (remember_facts_async returns List[str]) but only the first reached the toast, the memory event, and the panel highlight. Surfaced all of them.

- inference.py: MEMORY SSE event now emits data={'stored': True, 'fact': _memory_fact, 'facts': _memory_facts} (fact kept for backcompat).
- stream-chat-response.ts: parses the facts array (string-only filter), onMemory type gains facts?: string[].
- memory-events.ts: MemoryEventInfo gains facts?: string[].
- useChatMessages.ts: toast prefers facts list; shows 'Remembered: <first> +N more' when a turn stored multiple facts.
- MemoryTab.tsx: highlight prefers info.facts[0] ?? info.fact.
- Tests +5 frontend (facts passthrough, non-string filter, +N more toast, facts-array highlight, memory-events facts passthrough), backend source-contract test updated to the new event shape.
- Verify: tsc 0 errors, 61/61 affected frontend tests pass, 50/50 test_memory_service.py pass, py_compile clean.