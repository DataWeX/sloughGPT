# Fix session regenerate path + feedback public exports

**Status:** done
**Sprint:** 2026-09
**gh:** —

## Task
Option B: align regenerate consumers on `POST /chat/{session_id}/regenerate`
(introduced by aa8c973f), and export missing `domain.feedback` public API
symbols so session inspector (and other routers) stop ImportError-ing.

## Body
- Branch: `fix/session-regenerate-chat-path`.
- Regenerate lives only on ChatRouter (`/chat/{id}/regenerate`); no `/session` alias.
- Restore chat.regenerate contract: load history, "No session context found",
  "Model not loaded", thinking event, SSE tokens.
- Retarget: web chat/session controllers, mobile store, TS SDK, docs, all tests.
- Include unmerged feedback exports from 06b26835 (get_message_feedback + 6 more).
- Keep session inspector `domain.knowledge.get_knowledge_memory` fix.
- Fix tests/server/test_session_router.py (13 failures) by moving regenerate
  coverage onto ChatRouter tests.

## Verification
- pytest tests/server/test_session_router.py + new chat regenerate tests green
- ruff clean on touched files
- frontend/sdk path assertions updated

## Result
- Option B landed: all consumers use `POST /chat/{session_id}/regenerate`.
- `chat.regenerate` restores full contract (history, No session context, Model not loaded, thinking, SSE).
- `domain.feedback` exports 17 public symbols (incl. get_message_feedback + 6 from 06b26835).
- Session inspector knowledge import → `domain.knowledge`.
- New `tests/server/test_chat_router.py` (8); session suite 25 green (no regenerate alias).
- 113 related tests green; ruff clean. Branch: `fix/session-regenerate-chat-path`.
- Merged to main (see MERGED section).

---

## MERGED (2026-09-23)

- Branch `fix/session-regenerate-chat-path` merged to `main` with `--no-ff`.
- Post-merge: 113 related tests green; ruff clean.
