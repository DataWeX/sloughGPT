---
id: 20260809_023823_fix-slonet-backward-pass-broadcast-bug-test-tokenizerpy-fail
title: Fix SloNet backward-pass broadcast bug (test_tokenizer.py failures)
status: done
tags: slonet,training,autograd,testing
created: 2026-08-09T02:38:23.819377+00:00
---

Fix SloNet backward-pass broadcast bug (test_tokenizer.py failures)

Verified fixed 2026-08-09: SloNet backward-pass broadcast bug is resolved (test_slonet_broadcast.py + TestSloEngineLearn in test_tokenizer.py: 17/17 pass). The _mul backward now uses _broadcast_back (same helper as _add), fixing the (99,) vs (512,) grad-shape failures. slonet.py's only uncommitted change is the editor's in-flight SloLSTM.generate() addition — untouched.