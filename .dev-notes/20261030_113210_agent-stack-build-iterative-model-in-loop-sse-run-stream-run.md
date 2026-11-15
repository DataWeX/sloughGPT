---
id: 20261030_113210_agent-stack-build-iterative-model-in-loop-sse-run-stream-run
status: done
board: 20261030_113210_agent-stack-build-iterative-model-in-loop-sse-run-stream-run
title: "Agent stack build: iterative model-in-loop, SSE run stream, run persistence, run->training export, multi-agent routines"
tags: ["agents", "training", "core"]
sprint: ""
gh: null
---

# Agent stack build: iterative model-in-loop, SSE run stream, run persistence, run->training export, multi-agent routines

User go-ahead (confirmed twice). Scope: (1) true agent loop — model sees tool results and decides next step (today plan is computed once, inference only fallback agents.py:614); (2) SSE stream on execute route (gateway passes SSE, proven); (3) persistence for runs/steps behind existing store pattern; (4) export seam emitting run traces in trainer dataset format, source: agent-run; (5) phase 2 foundation: more stateful agents/bots running routines on well-structured core infra (leverage multi.py, TaskQueue/PGQ — no new framework). Constraints: extend existing domain/agents, ToolSpec stays single contract, no second loop copy, no router migration (audit card separate). Branch feat/agent-stack-loop.
