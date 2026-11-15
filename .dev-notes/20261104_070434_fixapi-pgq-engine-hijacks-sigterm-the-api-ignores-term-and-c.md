---
id: 20261104_070434_fixapi-pgq-engine-hijacks-sigterm-the-api-ignores-term-and-c
status: done
board: 20261104_070434_fixapi-pgq-engine-hijacks-sigterm-the-api-ignores-term-and-c
title: "fix(api): PGQ Engine() hijacks SIGTERM — the API ignores TERM and can only be SIGKILLed"
tags: ["api", "pugqeep", "infra", "shutdown"]
sprint: ""
gh: null
---

# fix(api): PGQ Engine() hijacks SIGTERM — the API ignores TERM and can only be SIGKILLed

Found 2026-11-04 while verifying the lazy-guard landing (card 20261101_144013) on a clean worktree at main: `kill -TERM <api-pid>` does NOT stop the server. It kept serving /health (HTTP 200) 8 minutes later, two TERMs later, and only `kill -9` ended it. `stop_latency_s` 25 s in `scripts/benchmark_startup.py` is the same thing seen through that script's escalation path, not a slow shutdown.

MECHANISM: `domain/infrastructure/_internal/pugqeep/engine.py::Engine.__init__` calls `self.install_signal_handlers()` unconditionally (line ~1235), which does `signal.signal(SIGTERM, self._handle_signal)` / same for SIGINT (~1862). `_handle_signal` only sets `self._running = False` and calls `self.stop()` — it never re-raises or calls the previous handler. Because the API constructs a PGQ engine during lifespan startup (`PGQ core engine created`), the engine's handler replaces the one uvicorn installed via `loop.add_signal_handler`, so uvicorn's graceful shutdown never runs. `/proc/<pid>/status` confirms the state: SIGTERM is in SigCgt (caught), not SigIgn/SigBlk, and the main thread parks in ep_poll.

MINIMAL REPRO (two uvicorn apps, same code, one difference):
  app = FastAPI()
  @app.on_event('startup')
  def s(): Engine()            # skip this line for the control case
  @app.get('/')
  def r(): return {'ok': True}
  -> control: exits 0.26 s after SIGTERM
  -> with Engine(): still alive after 10 s, killed -9

IMPACT: (1) every lifecycle hook registered for FastAPI shutdown is dead code in production — `_shutdown_process_guard`, `_cancel_parent_preload` (the new lazy-guard cancellation), checkpoint/flush/telemetry hooks: none of them run, because the only thing that ever ends the process is SIGKILL. (2) `systemctl restart slough-api` hard-kills after `TimeoutStopSec`, so in-flight requests die and no state is flushed. (3) Ctrl-C is equally hijacked, which is why dev-server stops need escalation.

FIX DIRECTION: a library must not install process-wide signal handlers from a constructor. Make installation explicit and owned: `Engine(config, install_signals=False)` by default, the standalone runner/CLI entry points call `install_signal_handlers()` themselves, and the handler chains to the previously saved handler after stopping the engine (or better: the engine exposes `request_stop()` and the owner does the signalling). Existing tests at `tests/core-py/test_pugqeep_process_mgmt.py:663,670,1433-1441` call `install_signal_handlers()` explicitly, so the API path can stop hijacking without touching them.

GATE: a test that boots the FastAPI app (or a minimal app + Engine()) and asserts SIGTERM ends the process within ~5 s and that the lifespan shutdown hook ran; plus a PGQ test asserting `Engine()` construction does not change `signal.getsignal(SIGTERM)`.
