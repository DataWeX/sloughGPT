---
id: 20261104_105716_fixtests-lstm-sampling-test-was-order-dependent-rng-created
status: done
board: 20261104_105716_fixtests-lstm-sampling-test-was-order-dependent-rng-created
title: "fix(tests): LSTM sampling test was order-dependent (RNG created and discarded)"
tags: ["tests", "flake", "determinism", "ci"]
sprint: "model-loading-v1"
gh: null
---

# fix(tests): LSTM sampling test was order-dependent (RNG created and discarded)

Found by CI/CD red on main `ef8c88bce` (run 41809319560, `test-core-py`), fixed in `da94fbf21`.

`tests/core-py/test_slonet_lstm.py::test_generate_sampling_does_not_collapse_to_single_token` opened with `np.random.default_rng(0)` and **discarded the result** — a no-op. `SloLSTM` initialises from the *global* `np.random` state, so the test's outcome depended on whatever the previous test in the file left behind: measured distinct-token counts over 4 repeats per seed — seed 0 → 2, seed 1 → **1**, seed 2 → 2, seed 3 → 2, unseeded → 1, 5, 1, 8. CI hit seed 1's case and the assertion (`>= 2`) failed.

Fix: `np.random.seed(0)` — the seeding the original line intended. Full file 70 passed on 3 consecutive runs, ruff clean.

**Same-push sibling red:** `test_domain_baseline.py::TestDirectoryHashDrift` flagged `slolib` + `training` after the torch removal (`0576757ef`). Fixed in `398698a20` (hash-only rehash; placement/classes/facets unchanged). Trap recorded: `scripts/domain_baseline_rehash.py` hashes **HEAD**, not the working tree — run it *after* committing, `--check` to verify. Running it before the commit prints "no drift" and CI reds afterwards.

Out of scope: the other 40-ish `default_rng(0)`-and-discard sites, if any — grep before touching, seed only where the global RNG feeds the assertion.

