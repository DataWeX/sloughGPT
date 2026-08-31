---
id: 20260808_030938_device-validation-explicit-cudamps-fall-back-to-cpu
title: Device validation: explicit cuda/mps fall back to cpu
status: done
tags: server,tests,models,device
created: 2026-08-08T03:09:38.518113+00:00
---

Device validation: explicit cuda/mps fall back to cpu

Fixed _resolve_device (apps/api/server/controllers/models.py:40) to validate explicit cuda/mps requests against actual availability. auto/None still delegates to auto_device() (mps>cuda>cpu). Explicit 'cuda' now checks _cuda_available(); explicit 'mps' checks _mps_available(); on absence it logs 'device=... unavailable - falling back to cpu' and returns cpu, so /models/load no longer reports a GPU that does not exist (the SloNet path is pure NumPy/CPU regardless, and only GGUF uses the device string via n_gpu_layers; the accelerator is auto-detected via slolib/gpu _detect_best_backend). Added 4 controller regression tests (cuda/mps available->stays, unavailable->cpu) + 1 router test (POST /models/load forwards device enum and echoes resolved device). Proven: new tests fail against the pre-fix pass-through. Live verified: device='cuda' and device='mps' on this GPU-less box both resolve to cpu with warning in log; server restarted (foreground nohup) and healthy. Full tests/server suite: 890 passed, 68 deselected (includes live integration tests).