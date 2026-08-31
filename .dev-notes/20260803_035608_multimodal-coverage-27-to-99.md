---
id: 20260803_035608_multimodal-coverage-27-to-99
title: Multimodal coverage 27% to 99%
status: done
tags: core-py,multimodal,testing
created: 2026-08-03T03:56:08.515687+00:00
---

Multimodal coverage 27% to 99%

Multimodal statement coverage raised 27% -> 99% via new tests/test_multimodal_coverage.py (~1200 lines, slow-marked). Fixed 2 source bugs found while testing: missing MultimodalManager._pick_seed_caption (added, stable seed from embed mean), AudioEncoder._embed_patches pads to max_patches+1. Engine: param_groups() is a method, not property; contrastive_loss.backward() sets no external grads; VisionCNN default embed_dim=128; browser-speech recognize is client-only (server guards on capabilities first). Coverage final: bpe_tokenizer 100%, char_tokenizer 100%, diffusion 100%, manager 100%, speech 100%, text_encoder 100%, tts 100%, video 100%, engine 99%, vae 98%, vision 98%. 8 remaining misses are unreachable defensive branches (all SloNet layers callable so else->forward() dead; embed() empty-token guard; beam EOS bookkeeping). Regression: 418 passed in 54s (coverage + slolib_gpu + v2 + generation + speech + tts). One flaky v2 sensitivity test passes in isolation.