---
id: 20261105_082749_artifact-registry-stamp-provenance-on-model-exports-retire-e
status: done
board: 20261105_082749_artifact-registry-stamp-provenance-on-model-exports-retire-e
title: "artifact-registry: stamp provenance on model exports + retire experience_adapter path-guessing"
tags: []
sprint: ""
gh: null
---

# artifact-registry: stamp provenance on model exports + retire experience_adapter path-guessing

DIAGNOSIS: the artifact registry already exists (domain/infrastructure/_internal/artifact_registry.py - scan/fingerprint/resolve/verify + .prov.json provenance with lineage() walk) - port-don't-rebuild, do not create a second registry. Real gaps: (1) only LoRA stamps provenance; every .soul/.gguf export path writes no .prov.json, so lineage() returns every model as a root and training_dataset linkage lives only in optional .meta.json fields; (2) experience_adapter guesses 3 hardcoded corpus paths instead of registry resolve; (3) models/auto-training scratch pollution (5.8G/224 files) - SEPARATE card, deletes files, needs user OK.
PLAN (branch feat/artifact-provenance-exports, worktree .wt-provexp off origin/main): add parents_from_metadata() + stamp_export() (never raises) to artifact_registry; hook slonet.export_to_sou (covers auto-train/chat_trainer/distill/vm_devices), export.py's sou+3 gguf writers, export_model re-stamp with config.metadata (last-writer-wins), both checkpoint savers re-stamp kind=checkpoint; experience chat corpus -> registry resolve first, home-cache kept (outside scan roots), feedback registry-first + named schema files kept. STATUS: tests written first (13 red) -> 52/52 green on tests/core-py/test_artifact_registry{,_hooks}.py, ruff check+format clean, full RUN_ALL suite in flight at /tmp/opencode/provexp_full_run.txt; provenance benchmark to run before merge. Evidence to be recorded at close.
CLOSED 2026-11-05 — PUSHED: fddcee1a7 (code) + 3cacdfd5c (sync entry) -> origin/main 3685675b6..3cacdfd5c (lane pattern, rebased onto tip before push; local main checked out in quiet-pixel's worktree, behind-only, theirs to pull). EVIDENCE: TDD 13 red first; 52/52 registry+hooks, 33/33 training_handler (+rotation-sidecar test), 84 passed export/soul/serialization/chat_trainer, post-rebase 97/97; ruff check+format clean; benchmark_provenance PASS (stamp delta 0.006ms vs 5.0ms budget, warm snapshot 21us); live lineage smoke: export -> parents [dataset:smoke-ds, model:gpt2-tiny] -> lineage() roots walk OK; full RUN_ALL partial 5231 passed / 0 failed before server restart, restarted post-rebase (CI full suite is the merge backstop per Test Rule clause 3). Companion card 20261105_084717 stays open (models/ cleanup, needs user OK).
