### tokenizer/BPE facade lane — verified this session
- suites: test_bpe_tokenizer.py, test_tokenizer_bpe.py, test_tokenizer.py, test_tokenizer_manager.py, test_tokenizer_router.py, test_slonet_tokenizer_manager.py, test_tokenizers.py (glob-truth on-disk, all exist at HEAD)
- result: 293 passed — RAW dots Read-verbatim from /tmp/tokenizer-facade-batch-20260923-071038.txt, exit=0
- committed-still-consistent with: 7ace6a55 (voice 5), c14283fb (tts 50), 63b424b6 (crossattention/fused_qkv/kv_cache 62)
- worker-gated/manual: domain/inference, domain/training, full core-py sweep (timeout)
- captured: 2026-09-23T07:13:03Z
