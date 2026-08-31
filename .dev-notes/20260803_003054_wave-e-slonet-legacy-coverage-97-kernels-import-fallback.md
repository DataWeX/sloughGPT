---
id: 20260803_003054_wave-e-slonet-legacy-coverage-97-kernels-import-fallback
title: wave E slonet legacy coverage: 97%, kernels import fallback
status: done
tags: slonet,coverage,tests,legacy
created: 2026-08-03T00:30:54.022815+00:00
---

wave E slonet legacy coverage: 97%, kernels import fallback

Wave E closes the slonet.py coverage gap from 96% to 97% (missing 31 -> 3). Appended +16 deterministic tests to test_slonet_legacy.py (now 243 passing, plus 775-line editor test_slonet_wave_a.py): accelerator old-backend failure fallback, Tensor dtype surface (fake-dtype to()/float() with float32 pin), point-weight cluster compression + set/get_point_weight, 2-layer LSTM forward adapter + 1-D skip_embed, attention-4D double-backward GQA accumulation, eval-mode batch-norm forward-grad, SoulLibBlock identity/no-params, non-callable _Layer, state-dict accelerator matmul/layernorm helpers, SO-variant import lineage (PK bytes -> slolib-pytorch), JSON tok_emb.weight key, params rebuild hidden_dim<=0 early return, numpy()-only state-dict branch, train_soul_transformer None-loss skip, plus subprocess test proving slonet imports with _KERNELS_AVAILABLE=False when slonet_kernels import fails (covers 51-52 behaviorally). Remaining 3 misses are non-measurable: 51-52 import-time except (only reachable pre-import), 107 dead loop in _broadcast_forward (line 104 normalizes ndim). Also un-ignored test_slonet_broadcast.py + test_slonet_compute_sensitivity.py in pytest.ini. Commits: f47ddb7.