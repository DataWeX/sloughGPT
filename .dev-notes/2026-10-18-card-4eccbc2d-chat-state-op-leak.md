---
id: 4eccbc2d
title: "chat: respond/stream own their state; fix ChatManager op-tracking leak"
status: done
board_id: Main
date: 2026-10-18
branch: fix/chat-state-op-leak
tags: [chat, state-seam, op-leak, qa, backend]
---

# Card 4eccbc2d — ChatManager op-leak + respond/stream state seam

Full card id: `4eccbc2d-c7bb-4b84-a86e-f68e776335a7` (board column is source
of the task text; this note carries Gene's rollout state).

## Dispatch (2026-10-18, Gene batch of 3)

- **Worktree**: `.wt-chatstate` (fresh cut from `main` @ f53190f7f)
- **Branch**: `fix/chat-state-op-leak` (commit only if verification proves a
  real gap on current main; otherwise evidence-only card)
- **Routed to**: qa-developer
- **Do NOT duplicate**: card 7aa803a1 (in_progress) owns the op-leak fix
  `40581c25c` on `feat/chat-state-seam-main`, awaiting merge. Port-don't-
  rebuild: if that fix applies cleanly to main, this card verifies and
  reports — it does not re-implement it.

## Scope decisions (from card text + main-state verification by Gene)

1. Fix (1) op-leak (`return` inside try skips `else`, `_untrack` never runs
   on success): PRESENT at main (`domain/chat/_internal/manager.py
   respond()`); fix already exists at `40581c25c`. Verify leak fails a
   regression test pre-fix, fix passes post-fix, and the branch still
   merges cleanly onto current main (`git merge-tree`).
2. Fix (2) architecture (respond/stream delegating to inference's stateless
   facades): premise does NOT hold at current main — `routers/chat.py`
   routes respond/stream through `domain.chat.get_chat_manager()`
   (stateful kernel); `respond()` persists via `_store`. Verify with
   commands; stream persistence is client-driven (`/session/{sid}/context`,
   card 7f64b8fa done). Out of scope: `routers/mobile.py` `handle_chat`
   (separate endpoint — record as observation only).

## Evidence (2026-10-18, Gene — verification complete, NO code change)

Worktree `.wt-chatstate` left pristine (0 dirty, 0 commits on
`fix/chat-state-op-leak`): port-don't-rebuild — both findings are already
resolved elsewhere, see verdicts.

### Clause (1) op-leak
- Leak CONFIRMED at main: `domain/chat/_internal/manager.py respond()` —
  `return resp` inside try, `else: self._untrack` (line ~204).
- Fail-before: `git show 40581c25c:tests/test_chat_manager.py > tests/…`
  then `pytest tests/test_chat_manager.py -q` on pristine main →
  **1 failed, 18 passed** — `test_respond_success_untracks_op`:
  `assert 'cm-test-…' not in active_sessions()` (the leak).
- Fix already exists: `40581c25c` on `feat/chat-state-seam-main`, owned by
  card **7aa803a1 (in_progress)**. `git diff f02006e0b main -- manager.py
  test_chat_manager.py` = EMPTY (files untouched since branch base) and
  `git merge-tree --write-tree --messages main feat/chat-state-seam-main`
  = **0 conflicts** → the fix merges cleanly onto today's main (the
  "fast-forward" claim is stale only because main moved 4 unrelated commits).
- Pass-after: fix applied verbatim → `pytest tests/test_chat_manager.py -q`
  → **19 passed**; `ruff check` on both files → clean.
- NOT ported to this card (would strand against 7aa803a1's merge).

### Clause (2) architecture premise — FALSE at current main
- `apps/api/server/routers/chat.py`: respond/stream/regenerate/cancel all go
  through `domain.chat.get_chat_manager()` (stateful kernel), never
  inference's `handle_chat` facades.
- Persistence: `manager.respond` calls `self._store(...)` before return;
  covered by existing `test_respond_returns_text_and_stores_history`.
- Stream history is client-persisted by design: frontend
  `apps/web/features/chat/hooks/useChatMessages.ts` `storeSessionContext` →
  `POST /session/{sid}/context` → `routers/session.py:68
  SessionCore.store_context`; the stale-snapshot bug was card 7f64b8fa (done).
- Seam intact: nothing outside `manager.py` touches `_track/_store/_untrack`.
- Observation (outside card scope): `routers/mobile.py:849,876` still calls
  inference's stateless `handle_chat`.

### Gates (worktree pristine main @ f53190f7f, PYTHONNOUSERSITE=1)
- `pytest tests/test_chat_manager.py tests/server/test_chat_router.py
  packages/core-py/tests/test_chat_domain.py test_chat_domain2.py
  apps/api/server/tests/test_inference_router.py -q -m "not slow"`
  → **171 passed** (0 failed).
- `pytest tests/test_chat_loop_e2e.py -q -m slow` → **6 passed**.
- `ruff check domain/chat/_internal/manager.py
  apps/api/server/routers/chat.py` → clean; `py_compile` → OK.

### Verdict for review
Both findings closed without new code: (1) fixed at 40581c25c — **recommend
landing via card 7aa803a1** (merge-tree proves 0 conflicts today);
(2) premise not applicable to main. Card ready to close unless the parent
wants the mobile `handle_chat` observation carded separately.

## Closure (2026-10-18, reviewed by parent session)

Reviewed and **closed to done — evidence-only card, zero commits** (by
design; port-don't-rebuild).

- Fail-before proof reproduced independently by the parent session in
  `.wt-chatstate` (pristine main f53190f7f): applying only
  `tests/test_chat_manager.py` from 40581c25c → **1 failed, 18 passed**
  (`test_respond_success_untracks_op` fails: `active_sessions` still holds
  the op); adding `domain/chat/_internal/manager.py` from the same commit
  → **19 passed**. Worktree hard-reset back to pristine afterwards.
- Fix lives at **40581c25c** on `feat/chat-state-seam-main`, owned by
  in-progress card **7aa803a1** (merge when main is quiet; merge-tree = 0
  conflicts as of today). This card does NOT re-implement it.
- Finding (2) (respond/stream statelessness) verified false at current
  main: `routers/chat.py` routes through `get_chat_manager()`; respond
  persists; stream history is client-persisted via `POST
  /session/{sid}/context` (card 7f64b8fa).
- Open observation (not this card's scope): `mobile.py:849` still calls
  stateless `handle_chat` — card separately if desired.
