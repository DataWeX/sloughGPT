---
id: 20261030_122933_agent-phase-2-background-runtime-factory-hierarchy-for-mass
status: done
board: 20261030_122933_agent-phase-2-background-runtime-factory-hierarchy-for-mass
title: "Agent phase 2: background runtime — factory + hierarchy for mass agent production"
tags: ["agents", "core"]
sprint: ""
gh: null
---

# Agent phase 2: background runtime — factory + hierarchy for mass agent production

User vision (decoded): agent loops powered in the BACKGROUND (not request-scoped); infra = a FACTORY + HIERARCHY system pattern for mass agent production after training. Build on: TaskQueue/Pool/PGQ (sync core, seam to async host), MultiAgentOrchestrator (hierarchy), AgentRunStore (persistence), the new execute_stream loop (3fed50cdf). Deliverable: routine runner that spawns/manages many stateful agents on the pool, each with bounded model-in-loop execution, status via SSE; lightweight bots must not always hit the slow model. No second loop — reuse Agent.execute_stream.
