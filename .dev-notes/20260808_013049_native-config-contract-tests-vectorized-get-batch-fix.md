---
id: 20260808_013049_native-config-contract-tests-vectorized-get-batch-fix
title: Native-config contract tests + vectorized get_batch fix
status: done
tags: 
created: 2026-08-08T01:30:49.616624+00:00
---

Native-config contract tests + vectorized get_batch fix

1) Added TestAutoTrainNativeConfig (4 tests) to tests/server/test_auto_train_e2e.py locking the frontend native-method contract: n_embed/n_layer/n_head/block_size/checkpoint_dir forwarded into /auto-train/start config, visible via /auto-train/status, schema defaults, and 422 validation. 14/14 e2e pass. 2) Editor wave (02:06-02:09) vectorized get_batch in chat_trainer.py + distill_gpt2.py (advanced indexing), smaller TrainerConfig defaults, eval 50->10 batches, GC every 100 steps. Vectorized path crashed on Python-list ids (TypeError). Fixed by converting ids via np.asarray inside get_batch (kept list attribute contract). 290/290 targeted training suites pass. NOTE: full tests/server suite is order/timing-flaky (pre-existing; failures vary run-to-run, recur with my tests deselected) - not caused by this work.