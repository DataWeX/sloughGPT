---
id: 20260810_062458_native-c-inference-engine-real-tokenizer-provider-wiring
title: Native C inference engine: real tokenizer + provider wiring
status: done
tags: inference,native-engine
created: 2026-08-10T06:24:58.336710+00:00
---

Native C inference engine: real tokenizer + provider wiring

Completed:
- NativeEngine now resolves the real model tokenizer: explicit tokenizer / hf_model_id / config _name_or_path / derived from .slnc cache path (models--X--Y -> X/Y). MorphTokenizer.from_pretrained is cache-only (no downloads).
- set_tokenizer(), _build_prompt() (prefers tokenizer.apply_chat_template, falls back to format_chat), _stop_ids() (tokenizer chat_stop_ids or config eos), _sample() masks ids beyond tokenizer vocab.
- Fixed bug: _stop_ids attribute shadowed the method (renamed to _stop_ids_cache).
- Added NativeEngine.from_slnc_file(path) classmethod (SLNCParser + tokenizer auto-attach).
- Wired into provider chain: setup_providers(native_slnc_path=...) registers native-c and makes it the default text provider (opt-in, feature flag unchanged). Fixed discarded-return bug (engine.from_slnc_file on singleton returned new engine; now captured).
- Tests: +12 (TestTokenizerWiring 9, TestNativeProviderWiring 2, TestRealModelEndToEnd 1 slow/integration with file+memory skip guards). 65 pass incl slow; provider_processors 46 pass; slonet_integration 23 pass.
- Verified real end-to-end on Qwen/Qwen2.5-0.5B-Instruct .slnc: load 12-20s, tokenizer vocab 151643, stop ids [50256,151643,151644,151645], coherent greedy output; ~1 tok/s CPU.
- ROADMAP updated; flag comment corrected. Next open step per user: train a native SloNet model now that the engine path is solid.