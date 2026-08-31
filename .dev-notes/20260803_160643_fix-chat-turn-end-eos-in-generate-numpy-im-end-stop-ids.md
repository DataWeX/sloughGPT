---
id: 20260803_160643_fix-chat-turn-end-eos-in-generate-numpy-im-end-stop-ids
title: Fix chat turn-end EOS in generate_numpy (im_end stop ids)
status: done
tags: inference,slonet,bugfix
created: 2026-08-03T16:06:43.426447+00:00
---

Fix chat turn-end EOS in generate_numpy (im_end stop ids)

Root-caused two defects: (1) 504 E_INFRA_TIMEOUT on /inference/generate = 120s middleware timeout vs ~4 tok/s CPU generation; (2) output rambled past turn-end replying 'Human:' to itself. Root cause of (2): generate_numpy stopped only at eos_token (50256) but chat turns end with <|im_end|> (151645), so generation continued. Fix: added extra_stop_ids param to generate/generate_numpy/generate_numpy_stream in slonet.py; added MorphTokenizer.chat_stop_ids() resolving <|im_end|>, <|endoftext|> (151643), <|im_start|>, + eos; threaded via defensive getattr through slonet_provider, model_worker, slonet_server (9 sites). Verified: generate_numpy stops at <|im_end|> ('Hello! How can I assist you today?<|im_end|>' vs pre-fix '...<|im_end|>\n<|endoftext|>Human: What is'); ProcessGuard->worker path stops at im_end; 3 regression tests added. 206+316+84 slonet/provider/tokenizer tests pass; 1 pre-existing unrelated quantized-int8 failure confirmed on stashed baseline. Requires server restart (pid 11793) to deploy.