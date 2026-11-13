---
id: f1816d61-8d58-4874-8f99-34f23564f1f0
title: "Core-py single-process full gate wedges at 89% (slonet thread leak + orphan fork pool)"
status: wip
board_id: gene-batch-20261010
date: 2026-10-10
branch: fix/core-gate-wedge-leaks
tags: [tests, infra, pytest, core-gate]
---

Card f1816d61 — batch gene-batch-20261010.

## Root-cause inventory (verified 2026-10-10)

1. Leaked daemon `_producer` threads: `domain/inference/_internal/slonet_provider.py:1724`
   spawns `threading.Thread(target=_producer, daemon=True)` inside chat_stream
   paths; `tests/core-py/test_slonet_provider_wave_i.py` exercises those paths
   and never stops the providers — threads accumulate in the pytest process.
2. Orphan idle fork children: our branch still has the BARE-fork `mp.Pool` in
   vector_backend (main's forkserver fix `72e8fae2f`/`e69915c1` + tracker
   follow-ups a0c0561bb, b6666ee44, f5ccd713a, 356d73b83, 13791d13d never
   landed here). Fork children ignore SIGTERM (inherited pytest handler) and
   idle as orphans; every later test then waits on the 600s pytest-timeout.

## Fix shape

- (a) teardown in wave_i file itself: stop/join producer threads per provider.
- (b) conftest fixture: track + terminate/join mp Pools at teardown.
- (b2) port vector_backend fork->forkserver from main (quote shas).
- Zone-sharding stays the documented workaround until the full gate proves
  clean; do NOT touch pytest.ini (committed config, other sessions dirty).

Must-not-touch: tests/core-py/{test_slonet_wave_a,test_training_queue,test_vm,
test_tokens_router,test_compat_track,test_download_manager,test_external_download,
test_slonet_legacy}.py — other session has them uncommitted.
