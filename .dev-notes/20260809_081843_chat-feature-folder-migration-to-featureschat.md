---
id: 20260809_081843_chat-feature-folder-migration-to-featureschat
title: Chat feature-folder migration to features/chat
status: done
tags: web,refactor,chat
created: 2026-08-09T08:18:43.971047+00:00
---

Chat feature-folder migration to features/chat

Moved chat module into apps/web/features/chat/ via git mv: components/chat/* -> features/chat/components/<folder>/, hooks/useChat*/useVoiceChat -> features/chat/hooks/, contexts/{ChatContext,ChatToolbarContext,ConvSidebarContext} -> features/chat/contexts/, app/(app)/chat/page.tsx -> features/chat/ChatPage.tsx (wrapper left at route, preserves uncommitted modelController work). Rewrote all alias imports (@/components/chat/X, @/hooks/useChatX, @/contexts/*) to @/features/chat/*.

Root-cause bug fixed in migrate_chat.py: add() loop used cwd-relative os.path.exists so test files never entered the move map when run from repo root (WEB-rooted now). Secondary bug: relative-import rewrite appended .tsx/.ts extensions (tsc TS5097) — stripped across 82 migrated files + fixed in script. A git reset had restored stale copies of the old tree alongside the new one; removed redundant old paths via git rm (0 residual @/components/chat refs).

Verification: npx tsc --noEmit 0 errors; npx vitest run 326 files / 3128 tests pass (2 benign post-teardown timer warnings in datasets/page.test.tsx, pre-existing); features/chat = 130 git-tracked files. Backups: /tmp/opencode/prerun2_041610.tgz.