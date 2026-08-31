---
id: 20260729_015526_fix-process-guard-psutil-unboundlocalerror-modelworkerproces
title: Fix process_guard psutil UnboundLocalError + ModelWorkerProcess default
status: done
tags: infrastructure,process-guard,bugfix
created: 2026-07-29T01:55:26.658615+00:00
---

Fix process_guard psutil UnboundLocalError + ModelWorkerProcess default


process_guard.py: split try/except for psutil import vs usage — when psutil is absent, _memory_mb() returns None instead of UnboundLocalError. model_worker.py: changed silent default model_cls_path to raise ValueError('requires either slnc_path or model_cls_path'). 7 pre-existing test failures fixed, 32 process isolation tests pass.