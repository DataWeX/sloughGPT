---
id: 20261101_113742_featprovenance-artifact-lineage-parent-pointers-codeplatform
status: done
board: 20261101_113742_featprovenance-artifact-lineage-parent-pointers-codeplatform
title: "feat(provenance): artifact lineage — parent pointers + code/platform provenance in artifact_registry"
tags: ["provenance", "lineage", "infrastructure", "adapters"]
sprint: ""
gh: null
---

# feat(provenance): artifact lineage — parent pointers + code/platform provenance in artifact_registry

GOAL: trace any core-stack artifact (adapter, checkpoint, model, benchmark run) back to the code+config+platform that produced it and to its parent artifacts.

GAP (found, not guessed): domain/infrastructure/_internal/artifact_registry.py already unifies artifact kinds (model/dataset/checkpoint/adapter/weight/download), fingerprints content, and reads lineage+base_model from model sidecars. It does NOT: (a) keep parent pointers or offer a reverse lineage walk, (b) record code provenance (git commit, dirty flag, python/platform), (c) carry ANY of this for .npz adapters — per_user_lora.aggregate_best_adapters already writes source_users[] into the npz but no commit/parent artifact ids.

SCOPE (append-only, read-only on existing data; no UI in v1; never touches model-weight hot path):
1. artifact_registry: ProvenanceRecord dataclass + provenance_snapshot() (commit, dirty, python, platform, hostname, config fingerprint) + lineage() reverse walker following parent_ids to roots (mirrors token_tree.decompose) + sidecar read/write helpers.
2. per_user_lora.aggregate_best_adapters: stamp provenance sidecar with parent_ids = source adapter artifact_ids + commit.
3. benchmark_results.record(): add platform+config fingerprint (already records commit + loadavg).
4. Tests: lineage round-trip, merge child traces to both parents, commit recorded, forward-only on pre-existing adapters.
5. Benchmark: provenance stamping adds <1ms to aggregate.
NOT IN SCOPE: backfilling pre-existing adapters (data never recorded), UI/page (needs a UX_FLOWS journey first).
