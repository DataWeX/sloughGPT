---
id: 20261104_071931_fixtests-domain-baseline-tree-hash-for-domaintraining-is-sta
status: done
board: 20261104_071931_fixtests-domain-baseline-tree-hash-for-domaintraining-is-sta
title: "fix(tests): domain-baseline tree hash for domain/training is stale — test-core-py red on main since 6b44763fd"
tags: ["tests", "ci", "domain"]
sprint: ""
gh: null
---

# fix(tests): domain-baseline tree hash for domain/training is stale — test-core-py red on main since 6b44763fd

CI/CD `test-core-py` on main: `1 failed, 42695 passed` (883f43433, 28 min run) —
  FAILED tests/core-py/test_domain_baseline.py::TestDirectoryHashDrift::test_dir_tree_hashes_match_head
  AssertionError: domain dirs changed since catalogue — re-review class/facets for ['training'] (drifted: [('training', '116e5c00', '4fd972fc')])

CAUSE: `6b44763fd` (train: own-inputs dataset adapter, card 20261103_121721) added `domain/training/_internal/own_dataset_adapter.py` (+115 lines) and did not rehash `domain/domain-baseline.json`. The gate compares each catalogued dir's stored tree hash against `git rev-parse HEAD:domain/<name>`, so any committed change to a domain dir reddens it until the baseline is refreshed with the review. Timeline confirms attribution: `test-core-py` was green on 35a648f00 (05:36) and red on 920993d48 (05:42), whose only domain change is 6b44763fd.

FIX: `python3 scripts/domain_baseline_rehash.py`, confirm the diff only refreshes `domain/training`'s hash (entry keeps its class/facets/consumers — a new internal adapter module does not move the dir's placement), then commit the baseline with the change and note the re-review in the commit message. The gate's own docstring says the hash exists so placement gets re-reviewed whenever a dir's contents change — the review here is 'still training, no facet change'.

VERIFY: `./scripts/python -m pytest tests/core-py/test_domain_baseline.py -q` green on a tree that contains 6b44763fd (a tree older than that commit passes and hides the drift).

SIBLING RED on the same main SHA: `test-root` fails 4 tests in `tests/test_docs_api_parity.py` for the stale `controllers_lines` claim (card 20261104_071349). Both are 'gate did its job, the landing skipped the bookkeeping' — worth a shared follow-up on whether these two gates run in the PR workflow for the paths they guard, instead of only in CI/CD after merge.
