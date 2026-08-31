---
id: 20260815_052906_chat-skeleton-stuck-after-knowledge-clear
title: Chat skeleton stuck after knowledge clear
status: done
tags: chat,skeleton,bugfix
created: 2026-08-15T05:29:06.735824+00:00
---

Chat skeleton stuck after knowledge clear

Fix: chat screen stuck on loading skeleton after knowledge clear.

Root cause: loadSession (features/chat/hooks/useChatSessions.ts) set sessionLoading=true then awaited sessionController.fetchMessages unboundedly. http-client retries with a fresh 30s timeout per attempt (up to 3 attempts), so a slow/offline backend left ChatScreen.tsx skeleton rendering for minutes and masking the cached conversation / empty state.

Fix round 1 (85248097): wrapped fetchMessages in withRemoteTimeout() (wall-clock 8s, Promise.race). On hang/reject loadSession falls back to locally cached messages and always releases sessionLoading via the existing catch/finally, so the conversation renders.

Fix round 2 (ea99e050, hardening): withRemoteTimeout now takes a task(signal) and creates an AbortController that aborts on timeout, so orphaned requests are actually cancelled (http-client forwards opts.signal to its fetch call) while the race still bounds the await. loadSession records sessionIdRef.current = sessionId at start and adds stale guards: a load superseded by navigation to another session no longer applies merged/fallback messages, no longer overwrites CURRENT_SESSION_KEY/draft/saved/toast state. sessionController.fetchMessages(id, opts?) forwards { signal, silent } to apiGet only when provided, preserving the 2-arg path. Remote merge runs silent:true so best-effort merge never raises an error banner. http-client now treats a caller-provided abort as terminal (no 2 no-op retries, error surfaces as the abort, not a false timeout).

Fix round 3 (a8c82008, skeleton-release regression): round 2 made a stale load skip releasing loading, but the New Chat path (useChatMessages.newChat) mutates sessionIdRef without calling loadSession, so a stale s1 load could leave sessionLoading=true forever - skeleton stuck again. Fixed by tracking in-flight loads per hook instance (inFlightLoadsRef) and releasing sessionLoading only when the count reaches zero, regardless of which session the load targeted. Stale guards still prevent applying the wrong session's messages/state.

Fix round 4 (2d52cb0a, deleted-session resurrection): deleteSession never changes sessionIdRef, so a load in flight for the deleted session passed the stale guards and re-applied its messages + restored CURRENT_SESSION_KEY after the delete cleared them - deleted conversation came back on screen and on reload. Track deleted ids in deletedSessionsRef and treat them like stale navigation in loadSession's guards (skip merge/fallback/KV/toast); saveSessionToStorage unmarks an id when a new message re-creates the session. Loading still releases via the in-flight counter.

Fix round 5 (this session, send-during-load clobber): ChatInput is only disabled while streaming (loading), not while sessionLoading, so a user could send a message during the 8s load window and the settling load would replace their fresh messages with the stale loaded snapshot (data loss). loadSession now captures baselineMsgCount = messagesRef.current.length at start and superseded() also returns true when the messages array has diverged (length changed, e.g. user sent/deleted a message mid-load), so the load backs off: no setMessages, no CURRENT_SESSION_KEY overwrite, no draft/toast, loading still released via the counter. useChatMessages passes its live messagesRef into useChatSessions opts.

Note: an external actor staged a revert of round 4 (code + tests + journal line) during this session; restored from HEAD before committing round 5.

Tests: useChatSessions.test.ts 17 (round 5 adds send-during-load test: messagesRef diverges mid-load -> setMessages not called, no KV restore, loading released). All 13 chat hook files (148 tests) pass; tsc clean for this change (only other-actor WorkflowCard.tsx error remains, uncommitted).

Round 5 (committed be32d563): loadSession captures baselineMsgCount at start; superseded() also returns true when messagesRef.length diverges (message sent/deleted mid-load) so a settling load backs off (no setMessages / no KV / no draft / no toast) while sessionLoading still releases via the in-flight counter. useChatMessages passes its live messagesRef into useChatSessions opts. +1 test; 17/17 useChatSessions, 148/148 chat hooks, features/chat 68 files / 901 tests pass.

Full-suite sweep (Aug 15): 4 files failed / 6 tests + 2 unhandled errors. Resolved: MessageActions.test.tsx failed all 28 on jsdom read-only navigator.clipboard (Object.assign throws getter error) — fixed via Object.defineProperty(configurable:true) + afterEach delete, committed 133c3d76. CustomErrorHandler / OutputComparisonCard / api-monitor-store failures are cross-file navigator pollution, pass in isolation. Remaining = other-actor in-flight lane, not touched: monitoring/page.test.tsx 1 test mid-iteration (keep previous data on partial fetchAll failure); tsc errors in new hooks/useTrainingPolling.ts + useTrainingStream.ts (missing ToastFn import); dev-log.ts:87 fetch().catch can throw when fetch is stubbed to return undefined.