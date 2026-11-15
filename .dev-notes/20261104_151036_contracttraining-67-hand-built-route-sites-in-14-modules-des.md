---
id: 20261104_151036_contracttraining-67-hand-built-route-sites-in-14-modules-des
status: done
board: 20261104_151036_contracttraining-67-hand-built-route-sites-in-14-modules-des
title: "contract(training): 67 hand-built route sites in 14 modules — descriptor projection wave"
tags: ["contract", "endpoints"]
sprint: ""
gh: null
---

# contract(training): 67 hand-built route sites in 14 modules — descriptor projection wave

Census 2026-11-04 (endpoint audit 20261103_075928): apps/api/server/training/ carries 67 decorator/add_api_route declaration sites across 14 modules (router.py 27, webhook_endpoints.py 9, jobs_api.py 8, control.py 7 …), mounted live via infrastructure/startup.py:939. Cohesive feature area — migrate as a wave to ToolSpec + create_router, shrinking handbuilt-routes-baseline.json as each module lands. Gate ratchets: no new hand-built sites may appear meanwhile.
