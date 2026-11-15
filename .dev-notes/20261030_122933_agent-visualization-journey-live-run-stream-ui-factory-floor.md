---
id: 20261030_122933_agent-visualization-journey-live-run-stream-ui-factory-floor
status: todo
board: 20261030_122933_agent-visualization-journey-live-run-stream-ui-factory-floor
title: "Agent visualization journey: live run-stream UI (factory floor view)"
tags: ["agents", "ux", "web", "area:web"]
sprint: ""
gh: null
---

# Agent visualization journey: live run-stream UI (factory floor view)

User vision: visualize the mass-agent factory working (like clear-code demo videos). Surface = SSE run events (run_start/tool_start/tool_result/final) already streamed by POST /agents/{id}/execute/stream + orchestrate. Page Rule applies: needs a named journey in docs/UX_FLOWS.md + page-baseline entry + journey-level test; designer involvement. Data source = AgentRunStore + live SSE; never show model names/tok-s to users.
