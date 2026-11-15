---
id: 20261005_pytest_ini_mirror
title: "Test config: 2 competing pytest.ini files (card d2803a3c-5) — inner mirror drifts from root; duplicate basenames"
status: doing
date: 2026-10-05
tags: [tests, config, pytest, infra]
---

USER DIRECTIVE 2026-10-05: "continue building" (4th).

Picked the one open `todo` card in my lane where I already had direct
evidence: card `d2803a3c-5` "2 competing pytest.ini files + 55 duplicate
test basenames make r…". Confirmed unowned (no dev-note references it).

MEASURED (worse than the card claims):
- FOUR pytest config sources: `./pytest.ini`, `./packages/core-py/pytest.ini`,
  `[tool.pytest.ini_options]` in root `pyproject.toml`, and in
  `packages/aml/pyproject.toml`. (pytest prefers *.ini over pyproject, so the
  two pyproject sections are effectively dead where an ini coexists.)
- duplicate test basenames now **86** (card said 55): `test_cli.py`,
  `test_config.py`, `test_companion_router.py`, … across `tests/` vs
  `packages/core-py/tests/`.

ROOT CAUSE: pytest selects ONE configfile = nearest ini to the common
ancestor of the args. So behaviour depended on invocation directory. The
root ini even states the intent — "Mirror packages/core-py/pytest.ini so
rootdir flips don't drop these" — but the mirror had drifted.

| key | root | packages/core-py (before) |
|---|---|---|
| `--continue-on-collection-errors` | yes | **no** |
| `--import-mode` | `importlib` | default `prepend` |
| `addopts` | `-v --tb=short …` | `-q` |
| markers | unit/integration/e2e/slow | slow/integration only |
| `filterwarnings` | 6 rules | none |

PROOF THE DEFECT IS REAL (not hypothetical): built a probe file that
fails to import and ran it next to a healthy file, rooted at
`packages/core-py`:
- before: `Interrupted: 1 error during collection` → **0 tests executed**
  (46 otherwise-passing tests never ran)
- after (`a7f7d541a`): `46 passed, 2 warnings, 1 error`, exit **1**

So the fix surfaces the failure without nuking the run — exactly what the
root ini's own comment promises.

FIX COMMITTED a7f7d541a — mirrored root's behaviour into
`packages/core-py/pytest.ini` (+29/-3), keeping directory-relative
paths/testpaths and the inner `--ignore=tests/test_neural_e2e.py`.

VERIFICATION (no regression):
- packages/core-py collection: **43107 before and after**
- repo root collection: **45245 + 53 errors, unchanged**
- `--import-mode=importlib` vs `prepend` on a fixture-heavy subset:
  **identical** 9 failed / 124 passed → behaviour-neutral
- unknown-mark warning on test_slo_embedder_edge: **1 → 0**
- probe exit code stays **1** (failure not hidden)
- `tests/test_shared_indexes.py` 6 passed; `packages/avion` 368 passed /
  1 skipped

DELIBERATELY NOT DONE — still open on this card:
1. The **86 duplicate basenames** themselves remain. Dedup = renaming
   test files across two trees, much larger blast radius, deserves its
   own card. The importlib switch neutralises the collision risk meanwhile.
2. Two dead `[tool.pytest.ini_options]` sections (root pyproject,
   packages/aml) — harmless but confusing; removing them changes no
   behaviour and should be confirmed before touching.
3. Root's `.pytest-libs` pythonpath entry NOT propagated: its symlinks
   target `../.venv/...` which does not exist in this worktree, and
   playwright/greenlet/pyee all resolve from the conda env.

CAUTION for whoever continues: running the journey suites rewrites the
tracked artifact `packages/core-py/tests/test_results/user_journey_results.json`
(4290 → 2 bytes observed). Use `--collect-only` when probing.
