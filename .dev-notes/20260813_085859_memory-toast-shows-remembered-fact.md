---
id: 20260813_085859_memory-toast-shows-remembered-fact
title: Memory toast shows remembered fact
status: done
tags: memory,chat
created: 2026-08-13T08:58:59.724232+00:00
---

Memory toast shows remembered fact

MEMORY SSE event now carries the newly stored fact text. Core: KnowledgeMemory.ingest_from_chat() returns list[str] (auto_ingest_from_chat wraps it, unchanged contract); MemoryProvider.store_turn_facts(); MemoryService.remember_facts()/remember_facts_async(). Router emits data={stored:true, fact}; parser passes fact through onMemory; toast shows 'Remembered: <fact>' (truncated at 140 chars) with generic fallback. Tests: +4 knowledge, +4 provider, +8 service, +1 wiring, +2 parser, +1 pub/sub, +3 hook. 191 py + 54 web pass, tsc 0.