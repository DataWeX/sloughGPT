---
id: 20260806_141641_fix-toolcapability-missing-file-searchcitation-agent-tool-ru
title: Fix: ToolCapability missing FILE_SEARCH/CITATION (agent tool runner)
status: done
tags: agents,tools,test-fix
created: 2026-08-06T14:16:41.356655+00:00
---

Fix: ToolCapability missing FILE_SEARCH/CITATION (agent tool runner)

Fixed pre-existing test_agent_core.py failures (9 tests: 6 shown at maxfail + 3 more) found while verifying the model-load fix.

Root cause: domains/agents/__init__.py ToolCapability enum was missing FILE_SEARCH and CITATION members, and ToolRunner.execute() had no routing branches for them, even though _run_file_search/_search_files/_run_citation/_generate_citations methods already existed. Agent._plan_with_keywords() also mapped 'search' -> WEB_SEARCH (tests expect FILE_SEARCH) and had no citation branch.

Changes (domains/agents/__init__.py):
- ToolCapability: added FILE_SEARCH='file_search', CITATION='citation'
- ToolRunner.execute(): added FILE_SEARCH -> _run_file_search and CITATION -> _run_citation routing
- _plan_with_keywords(): 'search'/'find' -> FILE_SEARCH (was WEB_SEARCH); web/online/internet/browser -> WEB_SEARCH; cite/citation/source -> CITATION

Verification:
- test_agent_core.py: 50 pass (was 9 failures)
- test_multi_agent.py: 64 pass; test_agent_system.py + test_agent_run_history.py: 54 pass
- apps/api/server/tests: 295 pass; tsc --noEmit: exit 0
- Full web suite: 226 files / 2288 tests, 2287 pass + 1 flake (DatasetsPage empty-state waitFor 1000ms timeout, passes in isolation 8/8) -- relaunch had to cd into apps/web (vitest cwd was repo root -> mass 'document is not defined' failures)
- Server healthy on :8000