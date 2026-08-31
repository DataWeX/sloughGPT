---
id: 20260808_234225_sdk-coverage-100-complete
title: SDK coverage 100% complete
status: done
tags: sdk,testing
created: 2026-08-08T23:42:25.463070+00:00
---

SDK coverage 100% complete

Closed the final SDK coverage gaps. Full matrix: 384 tests pass, 100% coverage (1753/1753 stmts) across all 10 sloughgpt_sdk modules. Added tests: test_sdk_models_extra.py (BatchRequest/ChatRequest/GenerateRequest flags, ChatMessage.assistant, BatchResult.from_response); cache break branch in DiskCache._check_size; websocket empty-raw recv + sync/async StreamIterator wait loops; client generate_stream JSONDecodeError via generator.throw(), import_dataset_github/url with name; cli __main__ exec (health dispatch + ImportError guard with raising sys.exit); benchmarks load_test worker future-error (flaky perf_counter on 2nd call) + __main__ block. Key tricks: patched sys.exit must raise SystemExit or module exec continues past guard; perf_counter first call is start_time so flaky raises on call #2; dead except clause reachable only via generator.throw().