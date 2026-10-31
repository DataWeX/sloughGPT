---
id: 20260924_038_process_guard_hf_thread_generate
status: done
title: process_guard HF/thread generate crash fix
tags: process_guard,bug,backend
---
# process_guard._generate_fn/_stream_fn HF/thread fix

## Bug
_ThreadWorker._run_hf sets provider=None + _hf_model/_hf_tokenizer, but
_generate_fn/_stream_fn always use provider._tokenizer/_model → NoneType crash.

## Plan
1. Branch generate/stream on provider=None → HF model/tokenizer
2. Support generate_numpy + HF .generate() shapes
3. Torch-free tests exercising the HF branch
4. ruff + gate + merge main

## Completion

- Merged: `562a31fb` fix(process_guard): HF/thread generate uses _hf_model/_hf_tokenizer
- Gates: test_process_guard (52) + TestModelServerWithGuard (3) + scatter (5) = 60 passed; ruff clean
- Branch + temp worktrees removed; card 038 marked done.
