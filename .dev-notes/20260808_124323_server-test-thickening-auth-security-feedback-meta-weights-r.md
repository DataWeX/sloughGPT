---
id: 20260808_124323_server-test-thickening-auth-security-feedback-meta-weights-r
title: Server test thickening: auth, security, feedback, meta-weights, registry, metrics, config
status: done
tags: tests,server
created: 2026-08-08T12:43:23.067447+00:00
---

Server test thickening: auth, security, feedback, meta-weights, registry, metrics, config

Thickened 7 thin tests/server suites with edge/error/validation/method-restriction cases; all green.

Suites (old -> new):
- test_auth_router.py 26 -> 45: register field-exclusion + v1-hash persistence + overlong username/password 422 + wrong-method 405; login missing password_hash 401, failed-migration no-save, overlong password 422; token success audited + overlong key 422; /me invalid token 401, bad scheme, no password_hash leak; /verify payload response; /refresh bad scheme + 405.
- test_security_router.py 25 -> 31: slice-then-filter ordering, filter-after-slice semantics, newest-last ordering, 405s (PUT/DELETE/PATCH audit, PUT/DELETE keys).
- test_feedback_router.py 26 -> 39: workflow maps conversation_id->message_id/session_id, empty feedback_id fallback, overlong 422s, list default limit=50 + zero passthrough, update exclude_unset (args[1] == {"name"}), delete calls controller, 5 route-level 405s. FIX: TestRecordFeedbackWorkflow must patch 'controllers.feedback.get_feedback_controller' (workflow endpoint re-imports locally, line 38), NOT 'apps.api.server.routers.feedback...' - prior tests were hitting the real controller (~6s/test).
- test_meta_weights_router.py 27 -> 30: /meta-weights/ping ok + data shape + wrong-method 405 + no-manager-required.
- test_registry_router.py 27 -> 31: 405 variants (PUT/DELETE models, PUT/DELETE best, POST/DELETE stats).
- test_config_controller.py 25 -> 31: no-kwargs copy, unknown+None noop, config does not grow, negative values stored, no type validation (router's job), bool stored.
- test_metrics_router.py 23 -> 27: prometheus render error -> 500, JSON content-type, int start_time stringified.

Also completed this session's earlier target: test_experiments_router.py 25 -> 43 (data endpoint reads, status persistence, 405s, validation 422s; get_experiment accepts loose files - only existence checked).

Full tests/server suite (not slow): 1680 passed, 68 deselected, 60.5s. pycache cleared.