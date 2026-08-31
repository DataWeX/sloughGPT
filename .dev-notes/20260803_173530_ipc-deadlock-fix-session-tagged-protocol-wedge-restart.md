---
id: 20260803_173530_ipc-deadlock-fix-session-tagged-protocol-wedge-restart
title: IPC deadlock fix: session-tagged protocol + wedge restart
status: done
tags: infra,ipc,process-isolation
created: 2026-08-03T17:35:30.851922+00:00
---

IPC deadlock fix: session-tagged protocol + wedge restart

Root cause: abandoned streams leave stale resp_q data; worker feeder blocks in anon_pipe_write (full pipe), wedged-but-alive worker never restarted.

Fix:
1. Session-ID-tagged IPC: requests carry session_id, worker tags resp_q msgs (type, session_id, data); parent discards stale messages.
2. Bounded worker-side puts (_STREAM_PUT_TIMEOUT_S=30) so a full pipe cannot block the feeder forever.
3. Stall detection: parent raises WorkerStreamStalledError after _STALL_TIMEOUT_S=30 with no message; ProcessGuard restarts the wedged worker (shares restart budget, _restart_lock, _restart_worker/_restart_worker_locked).

Verified: test_process_isolation.py + related suites all pass. Live server restarted (PID 46351): /inference/generate 0.4-5s (was 30.001s), /health 17ms (was 504), Count-1-to-5 stops at 9 tokens. No errors in log.