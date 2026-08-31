---
id: 20260816_112203_tier-2-migrate-per-user-lora-metadata-store-from-sqlite-adap
title: Tier 2: migrate per-user LoRA metadata store from SQLite adapters.db to MogDB
status: done
tags: mogdb,migration,per-user-lora,feedback
created: 2026-08-16T11:22:03.958141+00:00
---

Tier 2: migrate per-user LoRA metadata store from SQLite adapters.db to MogDB

PerUserLoRAStore now auto-migrates a legacy SQLite adapters.db found inside the store root into the MogDB user_adapters collection (keyed by user_id). Existing MogDB records are preserved (journal authoritative); the SQLite file is removed afterwards. Added 2 tests: legacy row migration and keep-newer-MogDB-metadata. Verified live data/user_adapters journal intact (default user, feedback_count=299).