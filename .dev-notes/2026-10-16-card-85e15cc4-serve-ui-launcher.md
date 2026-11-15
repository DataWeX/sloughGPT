---
id: 85e15cc4
title: "Notebook serves the web UI: PGQ-Engine launcher (one entry point -> one URL)"
status: done
board_id: Main
date: 2026-10-16
landed: 22cb8d1c4, 2e09f1a70, dcc0165c0, d997173df, e883379bf
tags: [colab, pgq, launcher, notebooks, serve]
---

# Card 85e15cc4 — CLOSED: already landed (found at pickup, 2026-10-16)

**No code written by this session.** The card was picked up as "the one
buildable open card"; every scope item was already on main, landed by
Oluwatobi during this same session:

- `22cb8d1c4 feat(notebook): PGQ-Engine serve launcher — one entry point ->
  supervised API+web -> one URL (card 85e15cc4)` — the implementation
- `2e09f1a70 fix(serve_ui): engine.tree('serve') before spawn — dispatch
  needs a tree` — follow-up fix

## Gate results on current main (65f22d5eb, verified here)

- `python scripts/serve_ui.py --help` → usage printed, options
  `--ready-timeout` / `--no-web` (note: NO port-override flags; ports are
  fixed :8000/:5173 — see "pickup corrections" below).
- `pytest tests/test_sloughgpt_colab_notebook.py` → **11 passed**, including
  `test_serve_ui_launcher_present_and_help_runs` and
  `test_notebook_serve_section_uses_serve_ui_not_bare_popen`.
- All 4 scope items present: launcher (PGQ Engine consumer, fd-redirection
  logs `logs/serve_ui-{api,web}.log`, IPython signal restore), notebook
  cells 24-26 (markdown + `serve_ui.start()` + `serve_ui.stop()`), tests,
  docs (QUICKSTART.md:452, README.md:33-34 carry the URL).
- **Real-launch gate: PASSED (gate 5, main `225c32ec0`)** — :8000/:5173
  were occupied by another session's live stack (pids 1545/1547), so the
  gate ran on free alternate ports via the dev-stack env vars
  (SLO_API_PORT=8010 WEB_PORT=5183 — the landed CLI takes no port flags).
  Foreground launch: URL banner printed live to stdout, both services 200,
  INT stop → **both ports released, zero stray processes**, child-log
  banner counts unchanged (fd isolation intact).
- Final gates: ruff check+format green, `make colab-test` 11/11, battery
  (notebook + pugqeep engine/process-mgmt) **360 passed** in 77s.
- Merges: `4386a6e97`, `65f22d5eb`, `5165af794`, `620d7f460`, `225c32ec0`.

## Real-launch gate findings (gates 1-4 → 3 defects, all fixed)

1. **Dispatch needs a tree** — `Engine.dispatch()` routes ungrouped procs
   into an existing tree; with zero trees it returns 0 forever and children
   sit at CREATED (gate 1: 90s timeout, empty log). Fix `2e09f1a70`:
   `engine.tree(...)` before spawn — the engine tests always create one.
2. **Plain `Tree` runs fns INLINE in the launcher** — `SubprocessProcess`
   is created only by `GuardTree.branch`, which needs
   `EngineConfig(subprocess=...)` + `tree(guarded=True)` (the engine test's
   exact pattern). Inline meant `_open_child_log`'s dup2 rebound the
   LAUNCHER's fd 1/2 (URL banner landed in logs/serve_ui-api.log:114, gate
   file stayed 0 bytes), cancel was event-only while the fn kept blocking
   in Popen.wait, and kill -9 orphaned the Popen'd servers with ports still
   bound. Fix `d997173df`: guarded tree + a consumer-side
   `proc.on_cancel → sub.terminate()` bridge (pugqeep has NO cancel
   delivery to a forkserver child — `SubprocessProcess.terminate()` had no
   caller in any engine cancel path) + two-layer reap before reporting
   stopped.
3. **The API server swallows SIGTERM** — it creates `PGQ("sloughgpt")`
   during lifespan (`apps/api/server/main.py:114`) AFTER uvicorn's
   capture_signals, so pugqeep's `_handle_signal` consumes TERM/INT and
   never sets uvicorn's `should_exit`; group TERM was delivered and
   ignored, then terminate_grace=3 SIGKILL'd the wrapper and orphaned
   uvicorn (gate 4: :8010 still 200 after "stack stopped"). Fix
   `e883379bf`: wrapper forwards group TERM, bounded 5s wait, then
   SIGKILLs the group; `terminate_grace=10` outlives the escalation;
   stop() reap bound 12s. PGQ core untouched throughout.

## The screening lesson (carried forward)

This card survived a full queue screen as "unowned + no commit anywhere"
because the screen grepped commit MESSAGES for the card id. The implementer
named the id only in one commit subject, and my screen ran before
`22cb8d1c4` landed. The robust pre-claim check is **path-based**: for the
card's owned paths, run `git log main -- <paths>` and check the working
trees for dirt — a card whose owned paths were just rewritten is done or in
flight, regardless of commit-message grep. Applied to the next pickup.

---

# Original card text (for the record)

GOAL: `sloughgpt_colab.ipynb` currently starts ONLY the FastAPI server via
`subprocess.Popen(stdout=DEVNULL)` and never launches the web UI. Make the
notebook end with a clickable URL (http://localhost:5173), WanGP-style: one
entry point -> supervised processes -> one URL, readiness reported.

## Owned paths (per card)

1. `scripts/serve_ui.py` — single Python entry point, a PGQ Engine consumer:
   one Engine, two supervised subprocesses (uvicorn :8000, vite :5173),
   port-ready poll then prints the URL, captured stdout tailed to
   `logs/serve_ui-{api,web}.log`, stop via `cancel_all()`. Skip
   `install_signal_handlers()` under IPython (kernel owns SIGINT). Must stay
   event-loop-free: `engine.run_background()` / thread.
2. `sloughgpt_colab.ipynb` section Serve: markdown + one code cell calling
   `scripts/serve_ui.py`, prints the URL; stop cell calls `cancel_all()`.
   Colab branch: detect `google.colab`; forward the port or degrade to
   "API up, web UI unavailable (no Node)".
3. `tests/test_sloughgpt_colab_notebook.py` — cell contract + launcher
   exists/`--help` assertion.
4. docs: QUICKSTART.md / README.md Google Colab line → the URL.

MUST NOT TOUCH: `scripts/compat/matrix.json`, `apps/api/server/routers/*`,
PGQ core (no event loop added there).

## Card claim corrections found at pickup (measured 2026-10-16)

- The card cites `packages/core-py/domains/infrastructure/pugqeep/engine.py`
  — STALE path. PGQ now lives at `domain/infrastructure/_internal/pugqeep/
  engine.py` (same `Engine` API: `run_background` :1483, `run_subprocess`
  :1649, `cancel_all` :1768, `install_signal_handlers` :1854). Tests import
  `domain.infrastructure._internal.pugqeep.engine`. Engine re-exported via
  the package `__init__` (`from .engine import Engine as Engine`).
- `scripts/serve_ui.py` does not exist anywhere (verified incl. `git log
  --all -S serve_ui` — the two hits are unrelated card text).
- Ports :8000 and :5173 were OCCUPIED at pickup (another session's live
  stack, pids 1545/1547) → the landed launcher honors the dev-stack env
  ports (SLO_API_PORT / WEB_PORT — no CLI port flags, correction to the
  original plan's `--api-port/--web-port`); the real-launch gate ran on
  free ports 8010/5183.
- `sloughgpt_colab.ipynb` clean everywhere checked; 2 branches
  (`feat/aml-stream-loader`, `fix/board-hygiene-doctor`) carry unique
  notebook diffs — coordinate if either lands first.

## Status

CLOSED — `feat/serve-ui-launcher` fully merged to local main (last merge
`225c32ec0`); ruff + colab-test 11/11 + battery 360/360 + real-launch
gate 5 all green (evidence above).

---

## Handoff from session `feat/notebook-web-ui` (2026-10-16 ~09:40)

**Claiming `~/Documents/sloughgpt-api-std` momentarily** to ff-merge 2 commits
onto main (per AGENT_SYNC → Sync recipe): `bench(startup): --via
{uvicorn,serve_ui}` + the AGENT_SYNC entry. Neither path
(`scripts/benchmark_startup.py`, `docs/AGENT_SYNC.md`) is dirty in that tree;
its 17 uncommitted entries are untouched.

This session independently rebuilt the launcher before `22cb8d1c4` landed —
caught at the gate by a **path-based** check (`git log main -- scripts/serve_ui.py
sloughgpt_colab.ipynb tests/test_sloughgpt_colab_notebook.py`, not
commit-message grep). Implementation **discarded; yours kept** (including your
env-port decision over CLI flags — the benchmark uses the same mechanism:
`SLO_API_PORT` set before importing `serve_ui`). Three measured deltas you may
want, all reproducible:

1. **Reuse-ownership of a stack we did not start** — the piece env ports work
   *around* rather than solve: before spawning, ask
   `commands.dev._check_api_ready(port)` / `_check_web_ready(port)` (the same
   helpers `slo dev` uses). Healthy → reused: print its URL, spawn nothing,
   `stop()` never touches it. Merely bound → `_find_free_port(port+1)`.
   Measured live on the foreign stack (pids 1545/1547, still 200/200):
   `start()` returned in **0.48 s**, `stop()` owned 0 processes, both ports
   still serving. Without it, `serve_ui.start()` on a machine with a dev stack
   up at the default :8000 raises "address already in use" instead of printing
   a URL — the notebook's default-env failure mode.
2. **Graceful stop as an alternative to gate 4's TERM→KILL.** The swallowed
   SIGTERM is installed by the app's own `PGQ("sloughgpt")` during the uvicorn
   lifespan; re-arming `server.handle_exit` after app startup (wrap
   `app.router.lifespan_context`, ~10 lines) stops in **0.18–2.2 s with a
   clean uvicorn "Shutting down" → "Finished server process" flush**, instead
   of waiting out escalation. Yours to accept or reject — the benchmark's
   `stop_latency_s ≤ 14 s` gate measures whichever you choose (yours: 2.23 s,
   comfortably green).
3. **Landed here: the regression gate for this card.**
   `./scripts/python scripts/benchmark_startup.py --via serve_ui --record` →
   `time_to_health_s` / `time_to_ready_s` / `stop_latency_s` for your launcher
   vs the raw-uvicorn baseline. Same load window: health **3.01 s** baseline vs
   **3.89 s** launcher, ready 123.0 vs 123.9 (app warm-up dominates both),
   stop **25.11 s (baseline, →SIGKILL) vs 2.23 s (yours)**. Caveat it encodes:
   `serve_ui`'s child log **appends** (`open(path, "a")`), so any log-derived
   readiness/warning metric must read a private dir (the benchmark re-points
   `serve_ui.LOG_DIR`) or it measures a previous run — this bit me: fake 3.88 s
   "ready", honest 126.4 s.

Also discarded with the duplicate (re-request if useful): 3 extra tests
(reuse-ownership + spawn→probe→stop), QUICKSTART:41 and tests/README launcher
lines, and a `Stack` object API (`summary()/status()/logs`) vs your dict.
