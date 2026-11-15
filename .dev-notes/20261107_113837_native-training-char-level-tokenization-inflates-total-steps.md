---
id: 20261107_113837_native-training-char-level-tokenization-inflates-total-steps
status: done
board: 20261107_113837_native-training-char-level-tokenization-inflates-total-steps
title: "native training: char-level tokenization inflates total_steps (200 epochs -> 9.7M steps, 1739h ETA)"
tags: ["training", "slonet"]
sprint: ""
gh: null
---

# native training: char-level tokenization inflates total_steps (200 epochs -> 9.7M steps, 1739h ETA)

Evidence job_cbc6162a (2026-11-07): prepare_data char-tokenizes data/qt/corpus.jsonl -> 55,263,226 tokens with 5871-char vocab (log: Data: 55263226 tokens, 5871 chars), i.e. 1 token ~= 1 byte, no BPE. steps_per_epoch = len(train_data)//block_size//batch_size = 49,736,903//128//8 = 48,571; x200 epochs = 9,714,200 total steps -> ETA 1739h at 1.4 steps/s. get_batch samples random offsets (np.random.randint) so epoch is nominal volume-equivalence, not a pass. Fix directions: BPE tokenizer path for native jobs (4x fewer steps/epoch), sane epochs default (early_stopping_patience=5 stopped it at epoch 1 anyway), and/or surface a steps-based ETA sanity check in the form. Also block_size=128 chars of context is tiny.
