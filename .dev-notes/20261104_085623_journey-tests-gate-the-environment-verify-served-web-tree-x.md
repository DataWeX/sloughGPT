---
id: 20261104_085623_journey-tests-gate-the-environment-verify-served-web-tree-x
status: done
board: 20261104_085623_journey-tests-gate-the-environment-verify-served-web-tree-x
title: "journey tests: gate the environment — verify served web tree (X-Slo-Web-Tree) + browser binary, skip honestly instead of content-assert reds"
tags: ["tests", "ci"]
sprint: ""
gh: null
---

# journey tests: gate the environment — verify served web tree (X-Slo-Web-Tree) + browser binary, skip honestly instead of content-assert reds

Goal: journey/page-load tests (tests/core-py/test_user_journeys.py, test_comprehensive_training_journeys.py) currently assert against whatever localhost:5173 serves — which may be ANOTHER checkout's UI. Make the environment a gate, not a failure (same class as the testmon-2.2 fix).\n\nEvidence (2026-11-04): a full local run showed 58 failed / 103 passed. Mechanism: the vite on :5173 served /home/mana/Documents/Default Project/sloughGPT (X-Slo-Web-Tree: ceb3ed22cf318353, HEAD f374830f9 — a day stale + uncommitted avion edits) while the tests came from current main → keyword scans ('lora', 'resume+checkpoint') missed → assert False. Single tests and the TestTrainingResults class passed in isolation (5/5); CI ran the same files green on the same head (no playwright in CI deps). Separately, the earlier sweep's 192 errors were browser-build drift (installed 1148 vs module wants 1234) — playwright install fixed it locally, but nothing GUARDS it.\n\nScope:\n- Serve check: a session/module fixture fetches the X-Slo-Web-Tree header from SLO_WEB_URL and pytest.skip('served by foreign/stale tree <sha>') when it does not match this checkout (vite.config.ts already stamps the header; scripts/grab-port.sh reads it — same mechanism).\n- Browser guard: skip with a clear message when the playwright browser build for the installed module is absent (ms-playwright cache path), so a package upgrade without 'playwright install' yields skips, not 192 errors.\n- Optional: surface both via one 'environment gate' helper so future journey files import it instead of re-deriving.\n\nNot in scope: fixing genuine content failures — those must still fail.\nGates: the two journey files run green locally against this tree; a deliberately-foreign SLO_WEB_URL produces skips, not reds.
