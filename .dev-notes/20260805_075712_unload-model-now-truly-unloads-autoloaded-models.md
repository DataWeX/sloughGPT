---
id: 20260805_075712_unload-model-now-truly-unloads-autoloaded-models
title: Unload model now truly unloads autoloaded models
status: done
tags: api,model-registry,bugfix
created: 2026-08-05T07:57:12.615082+00:00
---

Unload model now truly unloads autoloaded models

Fixed: /models/unload skipped registry teardown for autoloaded models (startup registers directly with registry, never sets controller _current_model), so /health stayed loaded. unload_model now resolves active id from controller state or registry.default_id, unregisters, adds clear_providers() helper, resets server state. Added autoload-bypass test. Verified live: after unload, /health shows model_loaded:false. 7+24+92 tests pass, tsc clean, 41 frontend tests pass.