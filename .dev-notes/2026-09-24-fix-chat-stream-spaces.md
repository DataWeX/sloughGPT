---
id: 20260924_049_fix_chat_stream_spaces
status: done
title: Fix missing spaces in streamed chat (decoder strip + frontend per-chunk clean)
tags: [bug, chat, streaming, tokenizer]
---
# Missing spaces in streamed chat

## Problem
Streamed assistant text glued at word boundaries (first ~50 chars lose spaces, then spaces return).

## Root causes
1. Frontend ran `cleanStreamedContent` (strips `^\s+`) per chunk while `assistantContentLen < 50` — 6 call sites (chat-store.ts x4, useChatMessages.ts x2).
2. `SloBPE`/`SloUnigram` `decode()` `.strip()` dropped inter-token whitespace on single-token streaming decodes.

## Fix (commit e0548edad, merged main 9ec1df9f1)
- Frontend: `cleanStreamedContent` only at message start (`assistantContentLen === 0`); `stripAssistantPrefix` kept through the 50-char window.
- Decoder: strip only when >1 non-special token; single-token decode returns raw so whitespace survives chunk concatenation (`domain/training/_internal/tokenizer.py` SloBPE + SloUnigram).
- Tests: 4 new pytest cases in `test_tokenizer.py` (single-token space preservation, boundary space, multi-token strip still holds).

## Gates
- pytest test_tokenizer.py: 94 passed; ruff clean.
- vitest chat-utils.test.ts 24 + useChatMessages.test.ts 19 = 43 passed.
