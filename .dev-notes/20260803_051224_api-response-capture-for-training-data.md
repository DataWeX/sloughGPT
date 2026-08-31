---
id: 20260803_051224_api-response-capture-for-training-data
title: API response capture for training data
status: done
tags: api,training,data
created: 2026-08-03T05:12:24.806930+00:00
---

API response capture for training data

Captured API responses for training data. ConversationLogger core module (domains/infrastructure/conversation_log.py) appends exchanges to datasets/api_conversations/{corpus.jsonl,input.txt} — messages + dialogue formats, thread-safe, env-gated (MAN_CAPTURE_CONVERSATIONS=0 disables). Wired into all 4 generation paths: /inference/generate, /inference/generate/stream, /chat, /chat/stream. 9 tests pass. Live-verified all 4 endpoints append rows.