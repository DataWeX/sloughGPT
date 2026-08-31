---
id: 20260801_132748_agent-runs-dashboard
title: Agent runs dashboard
status: done
tags: agents,ui
created: 2026-08-01T13:27:48.205776+00:00
---

Agent runs dashboard

Added orchestration run history dashboard (agents page + backend).

Backend: new file-backed AgentRunStore (packages/core-py/domains/agents/run_history.py) persists run records to data/agent_runs/<run_id>.json (goal, status, tasks, completed/failed counts, response, error, logs, timestamps; max 200 runs pruned). Orchestrate SSE now records start/plan/tasks/task-completion/failure/complete/error; run_id exposed in SSE events. New endpoints: GET /agents/runs?limit= (list newest-first), GET /agents/runs/{run_id} (detail, 404 if missing). Routes registered before /{agent_id} to avoid path shadowing.

Frontend: agents-controller.ts adds listRuns()/getRun() + AgentRun type; IconClock added to NavIcons. agents/page.tsx adds Run History card (status dot, goal, task-count badge, timestamp, expandable detail with task list, result, logs), refresh button, auto-refresh after orchestration completes/fails, loading skeletons, empty state.

Tests: 14 new store tests (packages/core-py/tests/test_agent_run_history.py), 4 new router tests (tests/server/test_agents_router.py, CI-only: fastapi not installed locally), 2 controller tests, 5 page tests (agents-page.test.tsx). Verified: frontend 1836/1836, tsc clean, store+multi-agent+server-integration pytest green, py_compile clean.