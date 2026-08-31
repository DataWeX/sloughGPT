---
id: 20260808_040845_device-surfaced-through-detailed-health-frontend-snapshot
title: Device surfaced through detailed health + frontend snapshot
status: done
tags: server,web,health,device
created: 2026-08-08T04:08:45.454158+00:00
---

Device surfaced through detailed health + frontend snapshot

Completed the device-reporting chain. Backend: get_detailed_health now includes 'device' (resolved load device via _get_model_device), matching basic /health; gpu.device_type (accelerator backend) kept separate. Frontend: DetailedHealth type + LiveHealthSnapshot type + mapDetailedToSnapshot + SSE onHealthEvent all carry device (fallback null). 3 new backend tests in test_health_controller.py (detailed device present / equals resolved / None when no model). Frontend tests: useLiveStatus mapDetailedToSnapshot maps device cpu + defaults null; DiagnosticsCard.test fixture updated for required field. tsc clean; full web suite 2505 passed, 2 pre-existing BestCheckpointCard verdict-text failures (fail in isolation, unrelated). Live: /health and /health/detailed both device=cpu; earlier server-suite run 977 passed.