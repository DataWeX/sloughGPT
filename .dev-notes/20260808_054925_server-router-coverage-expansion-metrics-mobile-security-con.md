---
id: 20260808_054925_server-router-coverage-expansion-metrics-mobile-security-con
title: Server router coverage expansion: metrics, mobile, security, config, kb
status: done
tags: tests,server,routers
created: 2026-08-08T05:49:25.060312+00:00
---

Server router coverage expansion: metrics, mobile, security, config, kb

Thickened 5 thin server router suites (tests/server/): lbn
- test_metrics_router.py 14→23: POST 405s, exact data keys, uptime string, empty model_type fallback, render None/multi-line, collector 500
- test_mobile_training.py 15→36: list/export/bulk endpoints, mobile_train 400/422, auto-config updates + bounds, param validation
- test_security_router.py 16→25: 405s, missing event_type filter, extra-field passthrough, 500 paths, exact keys
- test_config_router.py 17→21: all-six-fields update, GET/PUT 500, exact key set
- test_kb_router.py 17→39: pagination, update/related 404+fields, batch ingest, ingest-url scheme/host blocking, auto_tag, label 422/500
Full tests/server run: 1146→1181 passed, 68 deselected. pycache cleared.