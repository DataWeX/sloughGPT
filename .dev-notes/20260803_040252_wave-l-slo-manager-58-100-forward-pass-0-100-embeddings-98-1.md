---
id: 20260803_040252_wave-l-slo-manager-58-100-forward-pass-0-100-embeddings-98-1
title: Wave L: slo_manager 58%->100%, forward_pass 0%->100%, embeddings 98%->100%
status: done
tags: coverage,inference,slo-manager,forward-pass,embeddings
created: 2026-08-03T04:02:52.840640+00:00
---

Wave L: slo_manager 58%->100%, forward_pass 0%->100%, embeddings 98%->100%

Wave L: slo_manager.py 58%->100%, forward_pass.py 0%->100%, embeddings.py 98%->100%.

- slo_manager.py: 2 unreachable except blocks in _scan_souls pragma'd (no-cover; _parse_soul_info swallows all errors internally). embeddings.py: 3 abstractmethod pass bodies pragma'd.
- tests/test_slo_manager.py: +16 tests — real _scan_souls (top-level .slo+.soul, souls/ subdir, repo core-py/models/souls resolution via fixture with teardown, duplicate-name non-overwrite), preference load-restore success + read-error + save-write-error paths, get_trait_weights (no soul full schema, soul personality overlay, metadata cognition/emotion overlay via hand-registered soul in reader-expected SOUL+len+meta layout, text .slo non-SOUL magic branch, metadata read-error, empty-personality skip, live-config override, config-exception fallback), module-level switch_soul/list_souls.
- tests/test_forward_pass.py: +11 tests — ForwardPassResult defaults/shape, ForwardPassable runtime_checkable +/-/signature, timed_forward model_name/time/result.
- Found latent quirk: get_trait_weights metadata reader expects SOUL+meta_len+meta layout (no version field) unlike _parse_soul_info's versioned layout; behavior preserved, not fixed in this wave.
- Verified: 81 targeted tests, slo_format/context_managers/soul_engine/knowledge/rag regression + wave-K consumer sweep + npu/tokenizer/quantum sweep all pass. No filesystem pollution (models/ fixture cleaned up).