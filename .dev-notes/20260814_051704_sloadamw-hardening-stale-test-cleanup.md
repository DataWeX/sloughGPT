---
id: 20260814_051704_sloadamw-hardening-stale-test-cleanup
title: SloAdamW hardening + stale test cleanup
status: done
tags: training,slonet,tests
created: 2026-08-14T05:17:04.424308+00:00
---

SloAdamW hardening + stale test cleanup

SloAdamW hardening complete: SloAdamW(SloAdam) with decoupled weight decay, full type hints, 27-test suite + 17/17 smoke. Stale torch-era export tests fixed (test_export.py + test_slonet_legacy.py green). Environment-dependent tests made deterministic: test_lora_eval_router test_baseline_only (explicit nonexistent adapter_path), test_train_pipeline corrupt-newest mtime (time.time()+1000), test_vm 66-prefix MOV (assert deliberate ModRM 8B/89), test_vm_cpu_exec test_66_push_pop (66 5B). Full suite verified green: 378 files across 5 foreground batches, all exit=0.