---
id: 20261104_053224_fixmodels-modelloader-try-load-soul-ignores-model-id-newest
status: done
board: 20261104_053224_fixmodels-modelloader-try-load-soul-ignores-model-id-newest
title: "fix(models): ModelLoader._try_load_soul ignores model_id — newest mtime wins"
tags: ["model", "inference", "v1"]
sprint: ""
gh: null
---

# fix(models): ModelLoader._try_load_soul ignores model_id — newest mtime wins

Found while scoping card 20261101_144013 (model loading infrastructure).
`domain/infrastructure/_internal/model_loader.py::_try_load_soul` picks the newest-mtime soul under `models/slonet-native/` instead of honouring the requested `model_id`, so a load for model A can serve model B. This is the wrong-artifact class, one layer off the boot path.
Fix: resolve the artifact by the requested id (and fail loudly when the id has no artifact) instead of a directory wildcard.
Gate: a test that requests model B while A carries a newer mtime and asserts B is what loads (fails on today's code).

---

## CLOSED 2026-11-04 — landed on main `a7769616c` (+ domain rehash `3d29832af`), AGENT_SYNC entry `5684cf491`

**The bug, measured before the fix.** `models/slonet-native/` holds 5 TinyShakespeare souls (65-vocab char-level) and the resolver took the newest-mtime file regardless of the request:

- direct call `ModelLoader()._try_load_soul("Qwen/Qwen2.5-0.5B-Instruct")` -> `success=True`, `model_id=Qwen/Qwen2.5-0.5B-Instruct`, `soul_path=.../tinyshakespeare_1788619000.soul` — a Shakespeare model reported as Qwen (wrong-artifact class).
- temp-tree probe requesting `run-b` while `run-a` carried the newer mtime -> `run-a.soul` returned (`WRONG ARTIFACT`).
- it was invisible in production only because boot still goes through `lazy_from_slnc`; flipping boot priority to `.soul` first (sibling card) would have served a char-level model labelled Qwen on every boot.

**Fix (identity, not order).** Resolution now reads the checkpoint's own header (`read_soul_identity`, 64 KB bounded prefix + `raw_decode`, ~0.2 ms vs ~1.3 s full header parse) and never uses mtime:

1. exact checkpoint stem (or `<stem>.soul`), with the `resolve()`-prefix escape check mirrored from `domain/training/_internal/checkpoints.py`;
2. `hash:<prefix>` or a bare hex prefix >= 12 chars -> `integrity_hash`; an ambiguous prefix is refused (candidates listed) rather than guessed;
3. `dataset:<signature>` -> recorded `dataset_signature`; equal-signature runs are interchangeable by definition, so the pick is deterministic by filename and the count is reported;
4. family ids `{sloughgpt, sloughgpt-v1, slonet, slonet-native}` -> resolves ONLY when every candidate shares `base_model` and the same NON-EMPTY dataset signature; otherwise refuses with `name (hash12, dataset, born_at)` for every candidate. A file literally named e.g. `sloughgpt.soul` wins over the alias.

Requests that are not soul lookups (HF ids contain `/`) return `None`, so `_try_load_slnc` runs unchanged (`gpt2` etc. untouched). Soul-looking but unresolvable ids return a failing `LoadResult` (loud refusal). `_try_load_soul` now honours `self.models_dir`, which is what makes it testable — the old module-level `_REPO_ROOT` forced the only soul test to write into the real `models/slonet-native` (that test, `tests/test_goal12_owned_path.py::test_model_loader_prefers_soul_over_slnc`, was rewritten against a temp `models/slonet-native` + monkeypatched `get_model_dir` and still asserts soul-over-slnc priority).

**Verified after the fix (real tree, `models/slonet-native/`):**

- `Qwen/Qwen2.5-0.5B-Instruct` -> `None` in **0.0 ms** (falls through to `.slnc`; the Qwen `.slnc` on disk is still what boots).
- `sloughgpt` / `sloughgpt-v1` -> refusal in 5-27 ms listing all 5 checkpoints with `dataset=unrecorded`.
- `tinyshakespeare_1788619000` -> success, `matched_by=name`, `integrity_hash=692702867bf1ccb1`.
- bare `692702867bf1ccb1` -> success, `matched_by=hash`.
- family alias with two same-signature runs -> success, deterministic pick, `candidate_count=2`.

**Tests.** `tests/core-py/test_model_loader.py` **32 passed** (9 new: requested-not-newest, Qwen case, unknown-name fall-through, hash prefix, dataset signature, alias refusal on differing signatures, alias refusal on unrecorded signature, alias success on equivalent runs, path escape, ambiguous hash prefix, named-file-beats-alias). Affected suites (`test_models_init`, `test_hf_model_worker`, `test_startup_routers`, `test_slo_format`, `test_slo_format_extended`, `test_slo_format`+`test_checkpoints`, `test_soul_engine`, `test_direct_loader`, `test_training_checkpoint_export`) **455 passed**; `tests/test_goal12_owned_path.py` 7 passed; ruff check + format clean; `domain_baseline_rehash.py --check` clean.

**Behaviour change to know about:** `SLO_AUTOLOAD_MODEL=sloughgpt` (and `sloughgpt-v1`) now refuses instead of serving the newest soul, because all 5 real checkpoints record `dataset_signature=""` so equivalence cannot be proven. Follow-up card `20261104_081626` makes training record `SloProfile.dataset_signature` before `save_soul`; until it lands, request a checkpoint by name or hash.

**Not in scope (unchanged):** boot priority (sibling card), `models/auto-training` / `models/turbo-trained` search (that is `find_checkpoint`'s job), `_try_load_slnc`'s own wildcard, and family aliasing in the models UI.
