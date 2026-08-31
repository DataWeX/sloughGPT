---
id: 20260813_084733_chat-memory-feedback-sse-event
title: Chat memory-feedback SSE event
status: done
tags: frontend,memory,chat,backend
created: 2026-08-13T08:47:33.742826+00:00
---

Chat memory-feedback SSE event

Backend: moved remember_async out of post-gen gather, emits MEMORY SSE event (phase=MEMORY, data.stored=true) after the complete token in routers/inference.py. Frontend: stream-chat-response.ts now reads post-complete events (completed guard prevents double onComplete), adds onMemory callback; useChatMessages shows 'New fact saved to memory' success toast and stops loading at complete. Tests: 3 new parser tests, 11/11 pass; 15/15 hook tests; 41/41 memory service (incl. TestChatWiring contract); py_compile OK.