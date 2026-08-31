---
id: 20260813_050519_chat-context-inspector-tab
title: Chat Context Inspector tab
status: done
tags: frontend,context,chat
created: 2026-08-13T05:05:19.630161+00:00
---

Chat Context Inspector tab

Added ContextTab to ChatToolPanel (after Memory): steering modes + trait weights bars via soulsController, workspace memory counts + system prompt toggle via chatController.inspectContext() (new method GET /context/inspect). Additive section, all fetches fail-soft. 7 new ContextTab tests + 2 controller tests. 861 chat tests pass.