---
id: 20261103_074213_fixtests-test-rag-context-budgetpy-poisons-the-session-by-im
status: done
board: 20261103_074213_fixtests-test-rag-context-budgetpy-poisons-the-session-by-im
title: "fix(tests): test_rag_context_budget.py poisons the session by importing routers.inference under a second module path"
tags: ["tests", "isolation", "import"]
sprint: ""
gh: null
---

# fix(tests): test_rag_context_budget.py poisons the session by importing routers.inference under a second module path

Reproduced 2026-11-03 while landing 20261103_071231 (context steering).

`tests/test_rag_context_budget.py` inserts `apps/api/server` into sys.path and does
`from routers.inference import _RAG_CONTEXT_MAX_CHARS, _build_rag_block`. In the same
pytest session, `tests/server/test_inference_router.py` imports the canonical
`apps.api.server.routers.inference`. Those are two distinct module objects with two
distinct `_instance` singletons and two distinct copies of module state.

Symptom: running the two files together fails 5 chat tests with
`Model not loaded` (E_INFRA_REGISTRY) — `TestChat::test_chat_success`,
`TestChatPostGenTasks` x3, `TestChatStreamCancel::test_cancel_event_reaches_provider`.
Each passes alone, and each passes with every other file in the closure.

PROVEN PRE-EXISTING: checked out HEAD (bd7f342e5) for all six files and the same
5 fail. Not a regression from the context-steering change.

Fix direction (one line, needs the import to be canonical): drop the sys.path hack
and `from apps.api.server.routers.inference import ...`. Verify `tests/server` +
`tests/test_rag_context_budget.py` in one process is green, then check nothing else
relied on the `routers.*` alias. Why it matters: a full-suite CI run collects both
files, so this class of dual-import poisoning makes the chat suite order-dependent
and its reds unattributable.

FIXED same session (2026-11-03): dropped the sys.path hack, import is now
`from apps.api.server.routers.inference import _RAG_CONTEXT_MAX_CHARS, _build_rag_block`
— one module object per session, so `_instance` and module state are the router the
rest of the suite drives.

Evidence: `tests/server/test_inference_router.py` + `tests/test_rag_context_budget.py`
in one process: **5 failed / 82 passed -> 87 passed** (RAG tests unchanged and still
green, so no behaviour was traded for isolation).
