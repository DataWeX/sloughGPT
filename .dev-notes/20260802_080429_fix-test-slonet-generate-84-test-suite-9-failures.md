---
id: 20260802_080429_fix-test-slonet-generate-84-test-suite-9-failures
title: Fix test_slonet_generate 84-test suite (9 failures)
status: done
tags: slonet,tests,schedulers,eos
created: 2026-08-02T08:04:29.511909+00:00
---

Fix test_slonet_generate 84-test suite (9 failures)

Fixed 9 failing tests in packages/core-py/tests/test_slonet_generate.py (84/84 green).

Source fixes in packages/core-py/domains/training/slonet.py:
- generate/generate_numpy/generate_numpy_stream: eos_token default 0 -> None; _sample_from_logits eos mask no longer passed by these callers (mask unit-test kept)
- SloTransformer.lm_head property; _tie_weights uses it (fixes tie round-trip)
- SloNet._rebuild_from_state_dict drops tok_emb/lm_head layer appends (blocks + output norm only)
- _forward_state_dict tied-lm_head fallback sd.get('lm_head.weight', sd['tok_emb.weight'])
- LinearWarmupScheduler hold/no-decay branches return [base_lr]*len(base_lrs) not base_lrs

Test fixes: test_call_delegates_to_forward unpacks (logits, _); test_min_mode_reduces_after_patience loops range(4).

Regression: slonet suites 175/175; lr_schedulers/forward_pass/multimodal/train_pipeline/lora/distillation/rlhf/tts/performance all green; full core-py suite green except flaky test_pugqeep_cache_tasks::test_submit_training (passes in isolation). pycache cleared.