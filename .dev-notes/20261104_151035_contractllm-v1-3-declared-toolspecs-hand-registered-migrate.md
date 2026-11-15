---
id: 20261104_151035_contractllm-v1-3-declared-toolspecs-hand-registered-migrate
status: done
board: 20261104_151035_contractllm-v1-3-declared-toolspecs-hand-registered-migrate
title: "contract(llm_v1): 3 declared ToolSpecs hand-registered — migrate to create_router"
tags: ["contract", "endpoints"]
sprint: ""
gh: null
---

# contract(llm_v1): 3 declared ToolSpecs hand-registered — migrate to create_router

Census 2026-11-04 (endpoint audit 20261103_075928): routers/llm_v1.py declares CHAT/MODELS/EMBEDDINGS ToolSpecs (:470/:505/:516) but registers them via add_api_route (:562-580) — the descriptor exists yet the transport bypasses the projection, so auth/envelope/version are re-decided by hand. Done = all 3 routes emitted via create_router(spec, route, handler), runtime registry-vs-served check green, handbuilt-routes-baseline.json shrinks by 3 (reviewed edit).
