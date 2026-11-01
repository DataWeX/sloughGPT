Implement the loadable 'memory card' (asked 2026-08-27: 'build the memory card like a real memory card that can be loaded so the model remembers'). Spec exists in docs/FORMAL_SPECS.md; zero implementation. TDD save/load/restore lifecycle + model-load integration. [card 20260929_078, mined from stale sessions]

---

## Evidence (2026-10-10, close of card f74b9e75)

- **Landed sha**: `1f0cd0a03` feat(memory): loadable memory card —
  save/list/load/restore long-term memory (card f74b9e75)
- **Branch**: `feat/caret-core` (ancestor of HEAD). Not yet on `main`.
- **Gates**: `24 passed in 7.84s`
  - `tests/test_memory_card.py` — 20 passed
  - `apps/api/server/tests/test_memory_card_router.py` — 4 passed
- **Lint**: `ruff check domain/memory/_internal/memory_card.py
  tests/test_memory_card.py` — All checks passed
- **Invocation note**: the 2 router tests fail with
  `module 'apps.api.server.routers' has no attribute 'models'` unless
  `PYTHONPATH` includes `packages/mogdb/src` — `routers/models.py` does
  `from mogdb.cache import QueryCache` while `_instance = ModelsRouter()`
  runs at module import (models.py:99 / :1864). Pure invocation gap, not a
  defect: `main` carries the identical import. Incantation:

  ```
  PYTHONNOUSERSITE=1 PYTHONPATH="$PWD/packages/downcraft:$PWD/packages/core-py:$PWD/apps/api/server:$PWD/packages/mogdb/src:$PWD"
  ```

- **Why the card read `todo`**: this work was already synced to done by
  `64495146e chore(kanban): sync 6 delivered cards to done — memory card,
  provider config, token/sec, …`, and the done rows were then lost to
  stale-base board commits (`b17dfec2d`). Restoring the lost state rather
  than reporting new work.
