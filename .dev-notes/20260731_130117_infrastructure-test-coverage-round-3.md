---
id: 20260731_130117_infrastructure-test-coverage-round-3
title: Infrastructure test coverage round 3
status: done
tags: infrastructure,tests
created: 2026-07-31T13:01:17.484054+00:00
---

Infrastructure test coverage round 3

Round 3 complete: 312 tests green. New: test_model_catalog (29), test_output_buffer (38), test_download_manager (36), test_context_core (60). test_pt_loader removed — pt_loader is dead code (project does not load .pt; torch_load_checkpoint always falls back to torch.load due to map_location arg mismatch). Source fixes: download_manager._cache_dir returns Path (was str from get_cache_dir fallback, broke .exists()/.rglob()); context_core missing logger (NameError in get_context_core with MAN_VECTOR_STORE set); context_core used asyncio.get_event_loop().run_until_complete() (fails with no running loop) -> new_event_loop(). Verified test_auto_ingest (34) + test_training_pipeline (22) pre-existing green.