---
id: 20261112_062928_fixtest-datasets-importurl-x2-broken-by-origin-guard-ssrf-re
status: done
board: 20261112_062928_fixtest-datasets-importurl-x2-broken-by-origin-guard-ssrf-re
title: "fix(test): datasets ImportUrl x2 broken by origin-guard SSRF reject (tests/server, outside CI)"
tags: ["tests", "ssrf", "regression", "datasets", "ci"]
sprint: ""
gh: null
---

# fix(test): datasets ImportUrl x2 broken by origin-guard SSRF reject (tests/server, outside CI)

Found by the change-scoped closure of card 20261102_001441 (2026-11-12): tests/server/test_datasets_router.py::TestImportUrl::test_success + ::test_failure_400 are deterministically RED on origin/main since e9973ce4e (origin-guard merge). Three-strike evidence: red isolated (2.5s) AND red on pristine base (kernels edit stashed) AND diff-impossible for that card (its diff touched only tests/core-py/test_slonet_kernels.py). ROOT CAUSE: the origin-guard lane fixed the datasets SSRF helper swallow (except AppError: raise — its own AGENT_SYNC entry said "no test depended on the swallow"; these two DID): validate_url_not_private now actually rejects, and x.example is RFC-reserved NXDOMAIN, so slo.infra.ssrf logs "DNS resolution failed for x.example, blocking" -> 400 E_BAD_REQUEST BEFORE the mocked fetch. test_success expects 200; test_failure_400 expects error == 404 not found but gets the SSRF message. NOT CAUGHT BY CI: tests/server/ is outside the CI test scope (the documented CI --ignore), so the regression is invisible on main. FIX (small): retarget the fixtures off NXDOMAIN hosts per the origin-guard lane own precedent (a.example/b.example -> example.com variants in 3 test files) or mock the DNS resolution seam; keep the SSRF rejection itself intact (security fix must not be weakened). Evidence: /tmp/opencode/numba_run_affected.log (closure 7 reds triage) + AGENT_SYNC 2026-11-12 entry.

---

## RESULTS (done 2026-11-12, branch feat/datasets-ssrf-fixtures, pushed: 0b4c21f11 impl + 4580d9648 sync)

**Root cause confirmed exactly as carded** — watched RED first: `2 failed / 74 passed`, and the captured log is byte-identical to the card's diagnosis:
`SSRF check: DNS resolution failed for x.example, blocking` → `E_BAD_REQUEST [URL points to a private/internal address: https://x.example/data.txt]` → the guard fires BEFORE the mocked `URLImporter` ever runs.

**CI invisibility proven:** root job `--ignore`s `tests/server/` (ci_cd.yml:238); the server job runs `apps/api/server/tests/` instead. Main stayed green while this suite was red — the card's whole point.

**Fix (the card's named precedent):** both fixtures `x.example` → `example.com`.
- Matches the origin-guard lane's own `a.example`/`b.example` → `example.com` retarget, and `test_webhook_url_guard.py::test_accepts_public_url` which literally documents *"same DNS precedent as the datasets import tests"*.
- Verified `example.com` against the guard's EXACT predicate (`is_private_ip`): 4 resolved IPs (2 IPv4 + 2 IPv6), none private/loopback/link-local/reserved → passes. `x.example` → `gaierror` → blocked by design.
- Docstrings on both tests now state the resolvable-host requirement.

**MUST-NOT held — security pin ADDED (strengthen, never weaken):** `test_rejects_private_url_before_import` posts `http://169.254.169.254/latest/meta-data` (cloud metadata IP) and asserts:
1. `400`,
2. `"private"` in the error,
3. `URLImporter` **never constructed** — rejection happens before any importer work, no fetch attempted.

Non-vacuous: without the guard, `mock_cls` WOULD be called and status would be 200 → both asserts fail. As an **IP literal** it needs no DNS, so this pin cannot flake offline (unlike a hostname fixture).

**Gates:**
- RED watched: 2 failed / 74 passed (guard message byte-identical to card)
- GREEN: **77 passed** (74 + 2 retargeted + 1 new pin)
- Full `tests/server/`: **2134 passed, 1 skipped**
- `ruff check` + `ruff format --check`: clean
- Post-cherry-pick re-verify: 4/4 passed

**Lane hygiene:** first commit accidentally landed on the checked-out `feat/shell-flake-log-identity`; corrected immediately — that branch was reset to its already-pushed tip `fd9a0f417` (verified byte-identical to `origin`), and this work was rebuilt as an **independent lane** cherry-picked onto `origin/main` `46f3d8fdb`. No cross-card bleed; `origin/main..HEAD` = exactly 1 impl commit + 1 sync commit.

**No journal note exists for this card** (138 of 196 cards have none — card-only closure is the established pattern; creating one would risk a duplicate card since sync matches by title, not ID). Evidence lives in this description.
