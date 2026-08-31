---
id: 20260802_154011_slotransformer-no-soul-coverage-wave-d-94-slonet
title: SloTransformer no-soul coverage wave D (94% slonet)
status: done
tags: slonet,legacy,tests,coverage
created: 2026-08-02T15:40:11.816273+00:00
---

SloTransformer no-soul coverage wave D (94% slonet)

Added TestSloTransformerNoSoul (20 tests) covering soul_name='no soul' metadata, properties/tie_weights (+exception path), layer_norm norm fallback, forward use_cache/1d/list/cpu inputs, forward_pass, abs_pos_emb forward+generate, generate 1d/cpu, GQA no-kernel np.repeat, quantized no-kernel generate_numpy+stream (silu fallback 4086/4433), quantized kernel greedy stream (4473-4474), int8 lm_head fused argmax (4123/4467), load_state_dict shape-mismatch reporting + emb_drop alias + to/train/eval/context-manager. Fixed editor TestTensorFormatMethods expectations to match Tensor contract (float32 coercion, take_along_axis broadcast, view(*shape)). Coverage 92%->94% (166->111 missing, 203->188 branch-partial). 4564-4566 unreachable dead code in load_state_dict alt-name branch (lm_head.weight always in param_map).