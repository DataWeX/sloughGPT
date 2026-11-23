---
id: 20261103_164321_avion-exercise-loop-exerciserunner-has-no-production-caller
status: todo
board: 20261103_164321_avion-exercise-loop-exerciserunner-has-no-production-caller
title: "Avion exercise loop (ExerciseRunner) has no production caller — needs a journey before it lands"
tags: ["avion", "agent", "curriculum", "follow-up", "area:agents"]
sprint: ""
gh: null
---

# Avion exercise loop (ExerciseRunner) has no production caller — needs a journey before it lands

Parked on branch wip/avion-exercise-loop (commit 2b486ffd6, pushed) and backed up at ~/.qoder/tmp/inflight-20261103/change1-exercise-loop.patch. It started life in the same 10-file working diff as the two avion reliability fixes and was split OUT by user ruling (2026-11-03): Curriculum/ExerciseRunner are not exported from avion.ai and nothing outside tests constructs them, so by AGENTS.md step 14 it is a seam, not a journey. To land it: export the pair, pick the caller (CLI command or the agent run path), drive one practice batch end-to-end in the running app, and assert the journey rather than the render. The AgentEvent/_Hooks/on_event plumbing it adds is separately useful and already proven separable (40 tests pass without it).
