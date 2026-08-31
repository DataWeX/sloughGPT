---
id: 20260801_145428_profile-and-optimize-slonet-hot-loops
title: Profile and optimize SloNet hot loops
status: done
tags: agents,training,perf
created: 2026-08-01T14:54:28.093088+00:00
---

Profile and optimize SloNet hot loops

Profiled and optimized SloNet hot loops. All changes in commit 89994c4.

Optimizations (packages/core-py/domains/training/slonet.py):
- LSTM Tensor forward: precompute input-gate matmul `_matmul(xd, W_ih_T)` once for all timesteps, slice per step (was a matmul per timestep). Hoisted layer-2 input matmul out of the loop (h is loop-invariant). Forward now bit-exact vs forward_numpy (0.0 diff).
- `_slice` backward: fast `full[key] += g` path for basic indexing via new `_basic_index` helper; `np.add.at` only for fancy/overlapping indices. `bool`/`np.bool_` correctly treated as advanced.
- `_attention_4d`: replaced 6 einsum calls with BLAS matmul + transposes (forward + all backward grads), bit-identical output/grads (0.0 max diff) including GQA (n_rep>1) and masked paths.

Benchmarks:
- Transformer forward: 45.97 -> 24.62 ms/run (1.87x)
- `_attention_4d`: 7.23 -> 0.99 ms (7.3x)
- LSTM fwd+bwd: 15.26 -> 14.17 ms/run (profile script); 47.75 -> 38.0 ms/iter (seq32 h128 bench)

Verification: new packages/core-py/tests/test_slonet_lstm.py (9 tests: numpy parity, finite grads 1/2-layer, grad reference, training loss descent, _basic_index, slice grad). Full core-py slonet/training/multimodal/context/provider subset + 333 server tests + training router tests all pass.