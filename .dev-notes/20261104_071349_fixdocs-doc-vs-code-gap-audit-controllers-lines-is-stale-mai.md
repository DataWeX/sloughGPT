---
id: 20261104_071349_fixdocs-doc-vs-code-gap-audit-controllers-lines-is-stale-mai
status: done
board: 20261104_071349_fixdocs-doc-vs-code-gap-audit-controllers-lines-is-stale-mai
title: "fix(docs): DOC_VS_CODE_GAP_AUDIT controllers_lines is stale — main's test-root has been red since 289ca6686"
tags: ["docs", "ci", "tests"]
sprint: ""
gh: null
---

# fix(docs): DOC_VS_CODE_GAP_AUDIT controllers_lines is stale — main's test-root has been red since 289ca6686

CI/CD `test-root` on main failed on 883f43433 with `4 failed, 1618 passed` — all four are `tests/test_docs_api_parity.py`:
  FAILED test_synced_block_passes — AssertionError: ['controllers_lines: doc=2908 code=2918']
  FAILED test_claim_raised_by_one_is_an_alarm
  FAILED test_architecture_only_is_green
  FAILED test_architecture_only_json_reports_claims_equal_measured

CAUSE (not the commit that surfaced it): `289ca6686` (datasets: structured listing, card 20261104_060023) edited `apps/api/server/controllers/datasets.py` (+14/-4 = net +10 lines) without re-syncing the architecture claim in `docs/DOC_VS_CODE_GAP_AUDIT.md`. Verified by counting from git objects: controllers total lines 2925 at 920993d48 -> 2935 at 883f43433 (+10), and the measured claim difference is exactly +10 (2908 -> 2918). The gate's own comment above the block still says `Re-measured 2026-10-16: controllers_lines 2844->2908`.

FIX (one line): `docs/DOC_VS_CODE_GAP_AUDIT.md:76` `controllers_lines=2908` -> `2918`, and refresh the dated comment above it. No code change, no other claim moves (the CI failure lists controllers_lines as the only mismatch).

VERIFY: `./scripts/python scripts/check_docs_api_parity.py` reports `architecture contract: 11 claims, 0 mismatched, 0 unresolved` on a tree that CONTAINS 289ca6686 (a stale tree older than that commit passes locally and hides the problem), then `./scripts/python -m pytest tests/test_docs_api_parity.py -q` green.

PROCESS NOTE: this is the class of red the parity gate exists for — the doc must be re-measured in the same commit that moves the count. Two follow-on questions worth answering separately: (a) whether CI should run this gate on paths-filtered PRs touching `apps/api/server/controllers/**` (it runs in CI/CD only, so a landing can sit red for a while), and (b) whether the claim should be measured mechanically in the doc (generated block) instead of hand-maintained.
