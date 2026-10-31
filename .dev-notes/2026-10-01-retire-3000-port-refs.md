---
id: 20261001_retire_3000_refs
title: "Retire dead :3000 references (app lives on vite :5173)"
status: done
date: 2026-10-01
tags: [docs, cleanup, devops, ports]
board_id: e47e19ee-9be7-4e60-84a0-f7b3a3443fdc
board: e47e19ee-9be7-4e60-84a0-f7b3a3443fdc
hash: 86c17b69759082966b59460f9faca8674c404741f3a5b614909d85365a4bb9e7
---

# Retire dead :3000 references (app lives on vite :5173)

ss shows no listener on :3000 (curl → connection refused; browser tab gets 502).
Live stack: slough-web.service → vite :5173, slough-api.service → :8000,
slough-gateway.service → :8080. scripts/dev-stack.sh still prints
WEB_PORT=${WEB_PORT:-3000} while running vite (config port 5173);
apps/web/.env.local has legacy NEXTAUTH_URL=http://localhost:3000;
apps/web/.next/ leftover. Acceptance: grep clean for stale :3000 claims in
scripts/docs (or labels corrected to 5173), docs/engineering-overview.md port
table updated.

SHARED WORKSPACE
- index updated: no index change required (stale-claim/label corrections only —
  no new app/package/doc)
- shared copy reused: root toolchain only, no installs
- coordination named: none — no other card owns the :3000 references

## Resolution

MERGED to main 2026-10-02.

- Branch `fix/retire-3000-refs`, commit `34898f48` "docs(cleanup): retire stale
  :3000 web-port claims -> :5173 (card e47e19ee)" — 19 files, ±29 lines. Merged
  as a fast-forward: main `e1fbcee08` -> `34898f48`, worktree clean, tree
  byte-identical to the tested branch.
- Acceptance met: zero stale web-:3000 claims left in scripts/docs/READMEs
  (residual grep hits all categorized: historical records, another session's
  journey/testing territory, docker internal topology, behavioral fixtures,
  false-positive slices, Grafana's own port).
- Notables: engineering-overview.md port table said `Web :3001` (not :3000) ->
  corrected :5173; cypress.config.ts baseUrl was dead :3000 -> fixed :5173 (real
  e2e enabler); dev-stack.sh WEB_PORT default 3000 -> 5173; checks green
  (bash -n, ruff, JSON.parse).

### Deferred / follow-ups

(a) infra/docker container-port story needs an infra review card (compose
    `3000:3000`, nginx `server web:3000`, Dockerfile.vite healthcheck,
    infra/scripts/deploy.sh curl) — deliberately untouched.
(b) apps/web package.json `dev` script is still `next dev` (default :3000) while
    live slough-web runs vite :5173 — wiring beyond this card, candidate for its
    own card.
(c) journey/testing files (domain/testing, run_journey_tests.py, avion, core-py
    journey tests) keep their :3000 code defaults — owned by the journey session
    (coordination card e8fd60d4).

### Follow-up (2026-10-03, gate session)

(c) partially landed: `test_user_journeys`, `test_computer_use_training_integration`,
`test_e2e_training_trigger` repointed `:3000` → `:5173` with `SLO_WEB_URL`
override (commits `6b74a4024` + `de2fe6cf4`) — these were gate9's 118F
route-smoke class. Still `:3000` and untouched (journey session, card
`e8fd60d4`): `test_comprehensive_journeys.py`,
`test_comprehensive_trainer_unit.py`, `test_comprehensive_training_journeys.py`,
`test_ui_journey_chrome_devtools.py`, `test_ui_journey_generic.py`,
`test_ui_journey_library.py`, `domain/testing`, `run_journey_tests.py`, `avion`
(the 6 core-py files are slow-marked / non-gating so far).
