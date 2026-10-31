---
id: card088
title: "Card 088 — Router test gaps: model_stack + phoneme have no tests"
status: done
priority: high
created: 2026-09-29
branch: docs/api-doc-gap-audit
worktree: .wt-audit
source: scripts/audit_api_docs.py (card 083) router-test cross-reference
---

## Ask

Close the two router-test gaps the card-083 audit found: `apps/api/server/routers/model_stack.py`
(5 endpoints: GET /model-stack, POST /model-stack/{base,push,clear}, DELETE /model-stack/{name})
and `apps/api/server/routers/phoneme.py` (9 endpoints under /phoneme/*) had zero matching test
files — the only coverage-looking hit was `test_multimodal_router.py`, which exercises the
different `/multimodal/encode-phonemes` route.

## Done when

- `apps/api/server/tests/test_model_stack_router.py` and `test_phoneme_router.py` exist and pass.
- Re-running `scripts/audit_api_docs.py` reports `router modules / with tests : 56 / 56`.
- ruff clean.

## Evidence

Landed 2026-09-29 on `docs/api-doc-gap-audit` (worktree `.wt-audit`, uncommitted):

- `test_model_stack_router.py` — 7 tests: GET stack, push→remove, remove-missing
  (`removed: false`), duplicate push → 400 `E_BAD_REQUEST` (message sanitized in
  body; original only in server log), base swap preserves layers + chmod 444,
  clear-by-kind, clear-all.
- `test_phoneme_router.py` — 11 tests: engine mock injected into the singleton
  `PhonemeRouter` (via `router.routes[0].endpoint.__self__`, restored on teardown),
  encode/decode/score/batch/detect/synthesize/status/languages/visualize,
  422 validation (empty text / empty batch), engine failure → 500 `E_ENCODE_FAILED`.
- Full canonical server suite: **1206 passed / 0 failed**
  (`cd apps/api/server && pytest tests/`) — includes fixing 7 pre-existing
  `tests.test_support` import aborts + root-conftest downcraft shadow
  (details in card 083).
- `scripts/audit_api_docs.py` → `router modules / with tests : 56 / 56`.
- ruff clean.
