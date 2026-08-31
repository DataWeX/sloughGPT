---
id: 20260813_153822_turbo-training-ui-wiring
title: Turbo training UI wiring
status: done
tags: training,turbo,frontend
created: 2026-08-13T15:38:22.925595+00:00
---

Turbo training UI wiring

Backend: start-turbo + turbo/status in auto_train.py (7 router tests pass). Hook: useTrainingSession turbo polling mirrors shared fields. UI: new TurboCard + formatDuration helper, wired into /training below pipeline. Tests: 229 training-related frontend tests pass.