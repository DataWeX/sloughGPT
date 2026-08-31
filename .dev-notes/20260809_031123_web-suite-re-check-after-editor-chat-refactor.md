---
id: 20260809_031123_web-suite-re-check-after-editor-chat-refactor
title: Web suite re-check after editor chat refactor
status: done
tags: frontend,workflow,testing
created: 2026-08-09T03:11:23.650406+00:00
---

Web suite re-check after editor chat refactor

FINAL: Chat feature-folder migration to apps/web/features/chat complete and staged (133 files; components/chat, hooks/useChat*, useVoiceChat*, and old contexts removed; residual old-path imports 0; tsc clean; vitest features/chat 793 pass). Full web suite now fully clean: 325 files / 3134 tests pass, 0 unhandled errors (fixed post-teardown 'window is not defined' flake in app/(app)/datasets/page.test.tsx by adding afterEach(() => cleanup())). AGENTS.md active docs updated to features/chat paths. Concurrent session restored old tree from HEAD multiple times during session; final state reconciled and stable.