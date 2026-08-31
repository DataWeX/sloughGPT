---
id: 20260802_044038_kernel-layout-fix-cache-usage-patchability
title: Kernel layout fix + cache-usage patchability
status: done
tags: slonet,kernels,models,api
created: 2026-08-02T04:40:38.781249+00:00
---

Kernel layout fix + cache-usage patchability

1) Fixed _nb_fused_attention_multi/_single einsum broadcast crash for non-GQA SloTransformer when kernels ON: caller passed k/v as (new_len,H,E) but kernels/einsum expect (H,new_len,E). Added transpose branch (guarded step>0 or seq_len>1) at both call sites in generate_numpy and generate_numpy_stream. 3 new regression tests (force _KERNELS_AVAILABLE=True) confirm crash before, pass after; kernel matches no-kernel path exactly. 2) Made routers/models.py _hf_cache_dir a module-level global (was instance attr) so patch('routers.models._hf_cache_dir',...) works; fixes pre-existing TestCacheUsage 3 failures (15/15 pass now). Verified: 182 + 176 tests pass, live server healthy.