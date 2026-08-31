---
id: 20260813_145406_turbo-telemetry-async-polling-fix
title: Turbo telemetry + async polling fix
status: done
tags: training,frontend
created: 2026-08-13T14:54:06.118207+00:00
---

Turbo telemetry + async polling fix

Fixed startTurboTrain async contract: hook now polls GET /auto-train/turbo/status every 3s (mirroring startFineTune), surfaces live telemetry (progress/step/loss/speed/ETA/elapsed) into shared state, transitions to complete/error on terminal status. Added 7 backend router tests (test_auto_train_turbo_router.py) and rewrote hook turbo tests (running->complete polling, telemetry, poll-to-error, api error, exception). Frontend: 14/14 hook tests + tsc clean. Backend: 7/7. Commits: 6b3f9d67 (telemetry feature), fc02c686 (polling fix).