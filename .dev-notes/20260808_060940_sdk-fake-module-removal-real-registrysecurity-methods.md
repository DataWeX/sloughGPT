---
id: 20260808_060940_sdk-fake-module-removal-real-registrysecurity-methods
title: SDK fake-module removal + real registry/security methods
status: done
tags: sdk,api
created: 2026-08-08T06:09:40.418741+00:00
---

SDK fake-module removal + real registry/security methods

Deleted 5 fake SDK modules (billing/webhooks/dashboard/auth/registry). Added real methods for /security/keys and /registry/* (models, models/{id}, best, stats) to sync + async Python clients and TS client with StandardResponse envelope unwrapping. Rewrote CLI registry subcommand. Removed fake test classes + benchmark, added 8 py + 5 ts tests. Rewrote sdk-py README, added docs/API.md, regenerated stale egg-info PKG-INFO. Verified: 78 py tests, 70 ts tests, 1244 server tests, tsc client clean.