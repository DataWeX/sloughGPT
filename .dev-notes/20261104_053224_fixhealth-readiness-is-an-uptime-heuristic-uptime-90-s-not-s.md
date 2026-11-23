---
id: 20261104_053224_fixhealth-readiness-is-an-uptime-heuristic-uptime-90-s-not-s
status: todo
board: 20261104_053224_fixhealth-readiness-is-an-uptime-heuristic-uptime-90-s-not-s
title: "fix(health): readiness is an uptime heuristic (`uptime < 90 s`), not state"
tags: ["model", "v1", "api", "area:models", "track:now"]
sprint: ""
gh: null
---

# fix(health): readiness is an uptime heuristic (`uptime < 90 s`), not state

`/health` decides `model_loaded`/readiness from an uptime window rather than from lifecycle state: after scope B a server can be ready-but-materialising (provider installed, weights landing on the preload thread), and a fast restart reports not-ready while it is in fact serving.
Fix: explicit lifecycle state (loaded | materialising | degraded) fed by the lazy preload / `_preload_parent_weights`, with uptime remaining only as diagnostic output.
Gate: /health reports materialising during the preload window and loaded after it; a restart inside the window never reports ready-with-no-provider.
