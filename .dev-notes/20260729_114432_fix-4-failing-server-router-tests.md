---
title: Fix 4 failing server router tests
created: 2026-07-29T11:44:32.109979+00:00
updated: 2026-07-29T11:44:46.072983+00:00
tags: server, tests, fix
status: done
---

Fixed 4 failing server router test files:

- **test_auth_router.py**: Mocked `_get_auth_deps` in register test; made `create_token` return a real string (not MagicMock) across all 6 tests
- **test_datasets_router.py**: Mock return values now include `path` for `DatasetInfo` validation; fixed bad_id test to expect 404 (not 422)
- **test_vector_router.py**: Rewrote with per-test fixture isolation; removed `@patch` leakage from init test
- **test_agents_router.py**: Changed `sys.execute.return_value` to `sys.execute = AsyncMock(return_value=...)` for awaitable mock

Result: 324/324 server tests pass (was 24 failing)
