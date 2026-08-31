---
id: 20260803_043712_agents-domain-coverage-48-to-100-5-files
title: Agents domain coverage 48% to 100% (5 files)
status: done
tags: core-py,agents,coverage,testing
created: 2026-08-03T04:37:12.209465+00:00
---

Agents domain coverage 48% to 100% (5 files)

Completed the test-coverage wave for domains/agents.

Coverage before: 95% total (multi.py at 85%, 38 missed lines). Coverage after: 100% total, 882 statements, 0 missed.

Files at 100%:
- __init__.py (272 stmts) - Agent/ToolRunner/SecurityBoundary core via new tests/test_agent_core.py
- system.py (81 stmts) - AgentSystem CRUD + execute via tests/test_agent_system.py
- tools.py (146 stmts) - ToolRunner helpers via updated tests/test_tools.py
- run_history.py (127 stmts) - run recording/pruning via updated tests/test_agent_run_history.py
- multi.py (256 stmts) - MultiAgentOrchestrator via updated tests/test_multi_agent.py

multi.py additions (20 tests in TestLoadCustomAgents + TestUncoveredOrchestratorPaths):
- _load_custom_agents: valid config load, existing-key skip, default tools, malformed JSON, missing-key entry
- execute/async_execute empty-plan early returns
- _plan/_async_plan: dict-typed response, regex-embedded JSON extraction, regex-with-invalid-JSON fallback, string depends_on, unknown-agent defaulting
- _compose completed path, _generate text/error/plain branches, _async_generate plain-string branch

Patterns used: Path.home() monkeypatch for custom agent config; _FakeGenerate for _cmds injection; async fakes returning plain strings never async-coroutines (Agent._inference_fn is sync).

Regression: 647 tests pass (6-file multimodal suite + 5-file agents suite, -m 'slow or not slow'). Pre-existing env gaps excluded (fastapi/psutil/starlette). All changes uncommitted.