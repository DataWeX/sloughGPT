---
id: 20260809_194842_vm-console-training-launch-confirmation
title: VM console: training launch confirmation
status: done
tags: vm,console,frontend
created: 2026-08-09T19:48:42.361279+00:00
---

VM console: training launch confirmation

Training launch card now shows a dismissible inline 'Launched training job #N' confirmation when SYS_TRAIN_START returns EAX >= 1 (handleRun returns the result; handleLaunchTraining reads EAX from registers). Denied launches (EAX < 1) show no note. +3 page tests (55), +1 e2e spec (14), docs updated. Full suite 328 files / 3175 tests, tsc exit 0.