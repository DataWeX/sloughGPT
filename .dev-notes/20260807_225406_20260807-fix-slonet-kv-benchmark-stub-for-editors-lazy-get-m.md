---
id: 20260807_225406_20260807-fix-slonet-kv-benchmark-stub-for-editors-lazy-get-m
title: 20260807 Fix slonet kv_benchmark stub for editor's lazy _get_model
status: done
tags: 
created: 2026-08-07T22:54:06.293878+00:00
---

20260807 Fix slonet kv_benchmark stub for editor's lazy _get_model

Editor's slonet_provider.py change (lazy _get_model factory) broke _StubProvider contract in test_slonet_kv_benchmark.py TestStackCrossTurn (2 failures). Added _get_model() to the stub returning self._model; tiny_model is a real SloTransformer with new_kv_state(). Full file: 8 passed. Editor idle; core-py 2nd-wave re-verification fully green.