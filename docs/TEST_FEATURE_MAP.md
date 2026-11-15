# Test Feature Map — features as the test suite sees them

**Snapshot: 2026-11-05.** Compiled mechanically from a static parse of
`tests/` (AST + regex, no hand counts) plus the full `RUN_ALL` run's
`-v` output, plus `config/feature_flags.json` and the `FEATURE:` tag scan.
Purpose: input for the Test Rule clause 5 trim audit (card `299d8d38`'s
follow-up) — _which features is the suite spending its time on, and what
looks irrelevant?_

Scope note: Python tests under `tests/` are analyzed in depth; web/Vitest
suites are counted but not mapped here (separate follow-up).

## Top findings (read this first)

1. **One file is 8.6% of the entire Python suite.**
   `tests/core-py/test_shell_repl_more.py` alone holds **4,081 test
   functions**; the top 5 files hold **6,880 (14.7%)** of 46,649. Any
   lever big enough to matter starts here (parametrized-REPL coverage).

   | File                                     | Tests |
   | ---------------------------------------- | ----- |
   | `tests/core-py/test_shell_repl_more.py`  | 4,081 |
   | `tests/core-py/test_vm_os_layer.py`      | 1,200 |
   | `tests/core-py/test_vm.py`               | 657   |
   | `tests/core-py/test_vm_coverage_gaps.py` | 474   |
   | `tests/core-py/test_interactive.py`      | 468   |

2. **89 of 103 skips come from two shell files** — `test_shell_repl_more.py`
   (56) and `test_shell_tui_live.py` (33): TTY/live-terminal-dependent tests
   that did not execute in this environment. Environment-gated tests are
   legitimate, but a test that never runs anywhere headless is unverified
   code wearing a test's name — worth an explicit "runs where?" audit for
   these two files.

3. **17 same-subject test pairs across two trees** — `tests/test_X.py` and
   `tests/core-py/test_X__corepy.py` (e.g. `test_quantization` 25 vs 168,
   `test_knowledge_graph` 14 vs 102, `test_rag` 13 vs 52). **Checked and
   NOT proven duplicates:** same-name overlap across all 17 pairs is only
   28 tests (most pairs 0%), so name comparison clears them — but
   "different names, same input+assertion" is exactly the redundancy
   class, so each pair needs an assertion-level diff before either side is
   trimmed. The root-tree files are the `reusable-ci-core` CI subset
   (see `tests/README.md`), which is why both exist.

4. **Disabled flags carry no dead feature tests (good news).**
   `cloud_vector_store` and `native_c_inference` (both `disabled`) are
   mentioned only by `test_shared.py` and `test_feature_flags.py` — flag
   _registry mechanics_, not feature coverage. There is no army of tests
   defending features you turned off.

5. **`dpo` (`experimental`) is the largest experimental surface** — 38
   test files in `tests/core-py` mention it (84 across all of `tests/`).
   Experimental status + 38 files is a scope question for the product
   side, not a defect.

6. **The `FEATURE:` deletion guard protects almost nothing.** Only **6**
   source files carry a `FEATURE: name — description` tag (all 6 have test
   mentions), against 1,015 test files — the CI "Feature Tags" job guards
   a rounding error. Either grow the tagging habit or treat the gate as
   vestigial.

7. **4,799 test names appear in more than one file** — but the top names
   are generic (`test_defaults` ×292, `test_init` ×189, `test_custom` ×94).
   This is an **audit lead list, not a redundancy verdict**: same name in
   different classes usually tests different classes. Redundancy requires
   same input + same assertion (Test Rule clause 2); this map only tells
   you where to point the auditor.

## Scale and concentration

| Area                                       |     Files | Test functions | Share |
| ------------------------------------------ | --------: | -------------: | ----: |
| `tests/core-py/`                           |       826 |         42,736 | 91.6% |
| `tests/server/`                            |        68 |          2,163 |  4.6% |
| `tests/` (root)                            |       115 |          1,707 |  3.7% |
| `tests/contract/`                          |         6 |             43 |  0.1% |
| **Python total**                           | **1,015** |     **46,649** |       |
| Web/Vitest (`apps/web` + `packages/strui`) | 916 files | (counted only) |       |

## Feature-flag coverage

Status from `config/feature_flags.json`; count = test files mentioning the
flag string (mechanical substring match — includes registry tests).

| Flag                                                                                                                         | Status                   | Test files | Note                                               |
| ---------------------------------------------------------------------------------------------------------------------------- | ------------------------ | ---------: | -------------------------------------------------- |
| dpo                                                                                                                          | experimental             |         84 | largest experimental surface (finding 5)           |
| multimodal                                                                                                                   | enabled                  |         52 |                                                    |
| quantization                                                                                                                 | enabled                  |         37 |                                                    |
| kv_cache                                                                                                                     | enabled                  |         34 |                                                    |
| slonet_provider                                                                                                              | enabled                  |         21 |                                                    |
| model_server                                                                                                                 | enabled                  |         20 |                                                    |
| slonet_kernels                                                                                                               | experimental             |          7 |                                                    |
| semantic_cache                                                                                                               | enabled                  |          6 |                                                    |
| cross_attention                                                                                                              | enabled                  |          8 |                                                    |
| vlm                                                                                                                          | experimental             |          4 |                                                    |
| knowledge_memory / model_registry                                                                                            | enabled                  |    10 / 10 |                                                    |
| llm_nlp / on_device_training / cloud_vector_store / native_c_inference / process_isolation / soul_manager / session_kv_cache | experimental or disabled |     2 each | registry-level mentions only for the disabled ones |
| step_trace                                                                                                                   | enabled                  |          1 | thinnest enabled coverage                          |

## Test families by filename

`slonet` 65 files · `shell` 33 · `token*` 20 · `quant*` 15 · `lora` 12 ·
`multimodal` 11 · `export` 9 · `dataset` 9 · `server` 9 · `checkpoint` 8 ·
`dpo` 3. The `slonet_*` naming still reflects the old wave-by-wave build
(`wave_a`, `legacy`, `ops`, `network`, `compression`, …) — a consolidation
candidate by name alone.

## What was checked and NOT found (so nobody re-does it)

- **Dead tests for disabled features** — none (finding 4).
- **Untested `FEATURE:`-tagged features** — none; all 6 tags have test
  mentions (the tags are just few).
- **`__corepy` twins with identical test names** — only 28 same-name tests
  total across 17 pairs; names alone do not convict them (finding 3).

## Method

- Counts: AST parse of every `tests/**/test_*.py` (`def test_*` at any
  indent, `parametrize` marks counted separately), 2026-11-05 tree.
- Skips: grepped from the `-v` output of a full `RUN_ALL=1
scripts/run_affected.sh` run on `feat/artifact-provenance-exports`
  (base `origin/main` @ `3685675b6`).
- Flag mentions and `FEATURE:` tags: substring/regex scan of test and
  source trees, same date.
- Regeneration: re-run the same three scans; numbers are a dated snapshot,
  not a gate. If this doc is ever gated in CI, generate it — never hand-edit
  the numbers.
