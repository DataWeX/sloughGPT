---
id: 20260812_052653_auto-memory-layer-memoryservice-chat-loop-wiring
title: Auto-memory layer (MemoryService) + chat-loop wiring
status: done
tags: core,memory,auto-memory
created: 2026-08-12T05:26:53.984565+00:00
---

Auto-memory layer (MemoryService) + chat-loop wiring

Goal: auto-memory (self-indexing RAG) so KnowledgeMemory actually learns from conversations.

Built modular domain (packages/core-py/domains/memory/):
- memory_config.py: MemoryConfig, env-driven (SLO_MEMORY_ENABLED/MIN_CHARS/MAX_FACTS/STORE_PATH/SYNC), thread-safe singleton.
- memory_provider.py: MemoryProvider Protocol (store_turn/store/retrieve/stats) + KnowledgeMemoryProvider wrapping auto_ingest_from_chat / add_fact / search / stats. Store injectable for tests.
- memory_service.py: MemoryService facade (remember/retrieve/store/stats), fail-closed. get_memory_service() singleton. Producer-agnostic so the future task layer (option 3) can call the same API.

Wired remember() into chat post-gen in apps/api/server/routers/inference.py (asyncio.to_thread, guarded try/except, same pattern as learner ingest) - this closes the gap where KnowledgeMemory.auto_ingest_from_chat was never called by the chat loop.

Tests: packages/core-py/tests/test_memory_service.py (20 tests) - remember gating (empty/short/disabled), fact extraction + retrieval round-trip, store/dedup, stats, provider-level, singleton, and a source-contract test that the router wiring stays present. All 20 pass + test_knowledge_memory.py + test_knowledge_augmenter.py still pass.

Docs: added Memory section to docs/ENVIRONMENT.md.

Verification notes: fastapi not installed in this env, so test_inference_router.py cannot run locally; wiring verified via py_compile + the source-contract test instead.

Known gap (pre-existing, documented): chat_stream context-frame gate uses kmem.stats().get('total_items', 0) but stats() returns total_facts - frame/RAG layer stays skipped in chat; enrichment path is unaffected.

Session 2 (CLI + API + docs):

- Core: MemoryProvider protocol + KnowledgeMemoryProvider + MemoryService extended with list_all(limit) and clear() (fail-closed when disabled). test_memory_service.py now 28 tests.
- CLI: new apps/cli/src/commands/memory.py; memory group registered in apps/cli/src/cli.py (stats/list/search/store/remember/clear). Verified live via python3 cli.py memory --help and an isolated smoke run (real data/knowledge store untouched - one smoke fact added then removed, store restored to 149 facts). apps/cli/tests/test_memory_commands.py: 14 tests.
- API: new apps/api/server/routers/memory.py (MemoryRouter, prefix /memory: GET stats / list?limit= / search?q=&limit=, POST store / remember / clear) registered in routers/__init__.py (get_all_routers() now 39). POST bodies use pydantic StoreRequest/RememberRequest (missing field -> 422); search q Optional with explicit 400; list limit clamp 1..1000, search 1..100. tests/server/test_memory_router.py: 13 tests.
- Docs: Memory Router section added to docs/routers.md; memory CLI group documented in docs/integration/CLI_README.md (stale planned-shell doc - added a labelled current-CLI section).
- Verification: py_compile clean on all touched files; 28+14+13 memory tests pass; pycache cleared. Pre-existing uncommitted worktree (chat-loop wiring in inference.py, context_core RAG, token-tree CLI, training frontend test refactors) remains uncommitted.

Session 3 (read path closed - auto-memory feeds chat context):

- Closed the last loop: auto-memory was write-only in chat (remember() stored facts, but nothing read them back). ContextCore.build_context_frame memory layer was episodic-only; RAG layer used a relevance-gated enrich_with_knowledge (topical-overlap floor) that filters out personal facts.
- Added ContextCore.get_auto_memory_context(query, limit) - lazily imports get_memory_service, retrieves top-k facts via MemoryService.retrieve (offloaded via asyncio.to_thread), formats as [Memory] <fact>. Fail-closed on error; blank query returns empty. Class constant _AUTO_MEMORY_TOP_K = 5.
- build_context_frame memory layer now merges episodic + auto-memory content (source becomes episodic_store+auto_memory when auto facts present; budget cap retained; no layer when both empty; include_memory=False gates it).
- Tests: TestAutoMemoryInFrame in test_context_core.py - 9 tests (format, blank query, empty, fail-closed, frame inclusion, episodic+auto merge, no-layer-when-empty, budget, include_memory=False). context_core now 80 tests, memory_service 28 - 108 total pass.
- E2E verified: MemoryService.store() fact -> build_context_frame(query) -> memory layer contains "[Memory] The user prefers the code editor Zed over VS Code" (isolated tmp store).

Session 4 (sync_remember made meaningful - dead config fixed):

- SLO_MEMORY_SYNC / config.sync_remember was declared in memory_config.py but never consumed anywhere (grep confirmed). Chat post-gen in inference.py hardcoded asyncio.to_thread(get_memory_service().remember, ...).
- Added MemoryService.remember_async(user_message, assistant_response) - async variant that offloads remember() to asyncio.to_thread unless config.sync_remember is True (inline execution for tests/task runners).
- inference.py chat post-gen now appends get_memory_service().remember_async(user_msg or "", full_response) to the gather list instead of wrapping in asyncio.to_thread itself. /memory/remember endpoint stays sync (FastAPI runs sync handlers in its threadpool).
- Tests: TestRememberAsync in test_memory_service.py - 3 tests (offload-by-default verified via asyncio.to_thread spy checking __func__/__self__; inline-when-sync_remember verified via a to_thread that raises; false-when-disabled). Lesson: bound methods are recreated per attribute access, so assert calls[0][0].__func__ is MemoryService.remember, not `is service.remember`.
- Contract test TestChatWiring.test_router_invokes_remember_in_post_gen updated: now asserts "get_memory_service().remember_async" present in router src (was asserting .remember + asyncio.to_thread).
- Verification: py_compile clean; memory_service suite now 31 tests, context_core 80, CLI 14 + router 13 - all pass; pycache cleared. Worktree remains uncommitted (mixed with pre-existing token-tree/chat-loop/training refactors).
Session 5 (task-backed memory producer - SLO_MEMORY_STORE_PATH consumed):

- Made the documented-but-reserved SLO_MEMORY_STORE_PATH / config.store_path real: new packages/core-py/domains/memory/task_memory.py wires memory writes through the infrastructure InProcessTaskQueue (the "future persistent-task layer" the memory docstrings named).
- New task types: memory.remember (payload user_message/assistant_response, calls service.remember_async) and memory.store (payload content/topic/source, calls service.store). Both return {"stored": bool}.
- On success each handler appends one JSONL provenance record to <store_path>/facts.jsonl (default data/memory/) - durable, inspectable home for task-mined facts. KnowledgeMemory stays the retrieval index; the archive is provenance, not a second index. data/ is gitignored so the archive can't be committed.
- register_memory_handlers()/unregister_memory_handlers() mirror training_queue.py; wired into startup.py _phase_task_queue after training handlers. submit_memory_remember()/submit_memory_store() helpers enqueue Tasks (return task id).
- Tests: packages/core-py/tests/test_task_memory.py - 11 tests (store handler archives fact + defaults topic/source; duplicate/empty store no archive; remember mines turn + archives; short turn / disabled memory no archive; register/unregister; submit payloads + global-queue default; full InProcessTaskQueue end-to-end flow; unhandled type fails). Lesson: KnowledgeMemory(load_persisted=False) disables vector init so search returns []; use KnowledgeMemory() + clear_all() under the monkeypatched tmp paths (same as test_memory_service.py).
- Docs: docs/ENVIRONMENT.md SLO_MEMORY_STORE_PATH section updated (was "Reserved for future") - now documents the task-backed facts.jsonl archive.
- Verification: py_compile clean on all touched files; 31 memory_service + 80 context_core + 11 task_memory + 14 CLI + 13 router = 149 tests pass; pycache cleared. Worktree remains uncommitted.

Session 6 (task-memory bugfix round - 4 defects fixed, 102 memory-surface tests green):

- Whitespace-only content: KnowledgeMemoryProvider.store rejected only falsy strings, so content="  " stored True. Fixed boundary validation: `if not content or not content.strip(): return False`.
- submit_memory_remember/submit_memory_store NameError: called get_task_queue() but only imported Priority, Task. Added get_task_queue to the lazy imports.
- register/unregister_memory_handlers only targeted the global queue singleton, so an injected test queue ran tasks with no handlers ("No handler registered"). Added optional queue=None param (default global) matching the submit helpers' injection pattern.
- remember_handler test used a "prefers Zed" turn the extractor cannot mine (needs is/has/can/number pattern). Switched to the declarative "Gradient descent is the optimizer" turn.
- Docs: ENVIRONMENT.md Memory Behavior paragraph updated - memory layer merges episodic + auto-memory facts (get_auto_memory_context, top-k SLO_MEMORY_MAX_FACTS), not just prior-session episodes.
- Verification: 13 task_memory tests pass; 102 memory-surface tests green (31 service + 13 task + 14 CLI + 13 router + 31 knowledge); py_compile clean; pycache cleared. Worktree remains uncommitted.
Session 7 (memory.consolidate task - near-duplicate pruning):
- New `memory.consolidate` task: `plan_consolidation(facts, threshold)` (pure, in new `packages/core-py/domains/memory/consolidation.py`) groups facts by topic, union-finds pairs with n-gram cosine >= threshold (0.85 default, payload can override), keeps the longest fact per cluster, returns keep_ids/remove_ids/groups/removed_count.
- Store seam: `MemoryProvider.delete(ids)` protocol method + `KnowledgeMemoryProvider.delete(ids)` (delegates to existing `KnowledgeMemory.delete_by_id`, which frees content hashes) + `MemoryService.delete(ids)` fail-closed. `list_all` ids match vector-entry ids - verified delete works on them.
- `consolidate_handler(task)` in `task_memory.py`: list_all(limit=5000) -> plan -> delete; result {removed, kept, threshold}; appends a JSONL archive record (unconditional, even when removed=0). `submit_memory_consolidate(threshold, queue, priority)`; registered in `register_memory_handlers`; exported from `domains.memory`.
- Config: `SLO_MEMORY_CONSOLIDATION_THRESHOLD` (default 0.85) env + `MemoryConfig.consolidation_threshold`.
- Tests: 8 planner tests (`test_consolidation.py`), 4 delete + 2 consolidation-integration tests (`test_memory_service.py`), 6 handler/submit/queue tests (`test_task_memory.py`). Verified embedder behaviour: identical text 1.0, near-verbatim substring 0.845, paraphrase 0.586, cross-topic 0.16/0.0 - so the default 0.85 only merges near-verbatim copies (safe).
- Docs: ENVIRONMENT.md memory section - fixed stale SLO_MEMORY_STORE_PATH ("reserved for future" -> actual task archive dir), added SLO_MEMORY_CONSOLIDATION_THRESHOLD. Config module docstring table updated.
- Verification: 164 memory-surface tests pass (task 19 + service 41 + consolidation 8 + knowledge 31 + cli 45 + router ...), py_compile clean, pycache cleared.

