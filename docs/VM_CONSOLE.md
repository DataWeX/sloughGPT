# x86 VM Console

The VM console (`/vm`) is a browser-based x86-32 assembly playground backed by the
`X86VirtualSystem` in `packages/core-py/domains/shell/vm.py`. Programs run under an
RBAC role; training syscalls are gated behind the `ADMIN` role.

## Architecture

```
apps/web (VM Console)           apps/api /vm/run              packages/core-py X86VirtualSystem
┌────────────────────────┐      ┌─────────────────┐           ┌───────────────────────────────┐
│ source textarea        │─────>│ routers/vm.py   │──────────>│ X86Assembler / X86CPU         │
│ role selector          │      │ VMRunRequest    │           │ X86SyscallHandler (INT 0x80)  │
│ program samples        │      │ role → Role map │           │ X86RBAC permission gate       │
│ registers / VGA output │<─────│ VMRunResponse   │<──────────│ VMTrainingBridge → /training/*│
└────────────────────────┘      └─────────────────┘           └───────────────────────────────┘
```

## API Endpoints

| Endpoint                      | Method | Description                                                                                                                                                                                                                                  |
| ----------------------------- | ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `/vm/run`                     | POST   | Assemble + execute source. Body: `source`, `max_steps`, `role` (`user`/`admin`/`kernel`), `debug`, `keyboard_input`. Returns `VMRunResponse` (status, exit_code, registers, output, vga, memory_dump, `training_job_id`, `training_result`). |
| `/vm/builtins`                | GET    | Program catalog (hello, count, fib, sort, rainbow, primes, calculator, factorial, guess, train, train-status).                                                                                                                               |
| `/vm/info`                    | GET    | VM capabilities (ISA, register set, memory limits, features).                                                                                                                                                                                |
| `/vm/training/jobs/{id}`      | GET    | Bridge-tracked training job status (`job_id`, `api_job_id`, `status`, `progress`, `error`). Completed jobs also return the result JSON (`result`).                                                                                           |
| `/vm/training/jobs/{id}/stop` | POST   | Ask the API to stop a running job (proxies `POST /training/jobs/{api_job_id}/stop`). Returns `{status, job_id}`.                                                                                                                             |
| `/vm/session`                 | POST   | Start an interactive console session running the shell REPL. Body: `role`. Returns `session_id`, `role`, `status`. 429 `E_RATE_LIMITED` above 8 live sessions.                                                                               |
| `/vm/session/{id}/input`      | POST   | Deliver a line to the session keyboard buffer. Body: `text`. Returns `{accepted}`.                                                                                                                                                           |
| `/vm/session/{id}/stream`     | GET    | SSE stream: boot backlog, then live output (`phase`, `status`, `data.text`); ends on halt/error/close.                                                                                                                                       |
| `/vm/session/{id}`            | DELETE | Tear down a session immediately.                                                                                                                                                                                                             |

The `/vm/run` role mapping is `user → USER`, `admin → ADMIN`, `kernel → KERNEL`.
Console sessions apply the same mapping to their kernel role (RBAC gates the
`train*` commands the same way it gates `SYS_TRAIN_*`).

## Interactive Shell Session

The session console boots a shell REPL kernel (assembled guest code from
`apps/api/server/vm_builtins.py`, `get_builtin("shell")`) and drives it character by
character: each `POST .../input` pushes keys into the guest keyboard buffer, a pump task
steps the CPU, and output reaches the browser as SSE. Sessions live in memory in
`routers/vm.py` — up to 8 concurrent, evicted after 30 minutes idle — and each has its
own persistent-block-device filesystem that resets with the session.

| Command                                 | Description                       |
| --------------------------------------- | --------------------------------- |
| `help`                                  | List commands                     |
| `ls`                                    | Directory listing                 |
| `cat <file>`                            | Print a file                      |
| `cp <src> <dst>`                        | Copy a file (truncates dst)       |
| `write <file> <text>`                   | Create/overwrite a file with text |
| `echo <text>`                           | Print text                        |
| `uname`                                 | Kernel name/version               |
| `pid`                                   | Current process id                |
| `clear`                                 | Clear the screen                  |
| `about`                                 | Console banner                    |
| `halt`                                  | End the session (stream closes)   |
| `train`, `train-status`, `train-result` | Training bridge (ADMIN role only) |

Line editing is client-side: Up/Down recall history, Backspace sends `0x0E`. If the SSE
connection drops, the error banner's **Reconnect** button re-subscribes the same
session — the server re-sends the full scrollback as backlog (sequence-deduplicated),
so the session filesystem and history survive; only **Restart** (a new session)
resets them. The kernel has no `>` redirect (the keyboard path has no shift state) — use
`write`/`cp` instead; `rm`/`mv` await `SYS_UNLINK` in the VM. Unknown input replies with
the usage/unknown banner. Latency and throughput baselines come from
`scripts/benchmark_vm_shell.py`.

## RBAC Model

Three roles, defined in `vm_permissions.py`:

| Role     | Level | Grants                                                                    |
| -------- | ----- | ------------------------------------------------------------------------- |
| `USER`   | 0     | File I/O, process spawn, basic computation                                |
| `ADMIN`  | 1     | USER + process kill, device I/O (serial/mouse/disk/RTC/net), **TRAINING** |
| `KERNEL` | 2     | PID 1 only, unrestricted (all permissions)                                |

A syscall denied for the current role sets **EAX = -2** (`0xFFFFFFFE`) and does not
execute. The console detects this and shows a role-aware banner directing the user to
switch to the `admin` role.

## Training Syscalls

The training syscall surface is a thin bridge (`vm_training_bridge.py`) that proxies
guest calls to the existing `/training/start` and `/training/jobs/{id}` API — no
training logic lives in assembly or in the bridge.

| Syscall                | EAX | Args                                               | Returns                                                                 |
| ---------------------- | --- | -------------------------------------------------- | ----------------------------------------------------------------------- |
| `SYS_TRAIN_START`      | 28  | EBX = addr of null-terminated JSON config          | job_id (>=1), -1 on error, -2 if denied                                 |
| `SYS_TRAIN_STATUS`     | 29  | EBX = job_id                                       | 0 running, 1 completed, 2 failed, -1 not found, -2 if denied            |
| `SYS_TRAIN_GET_RESULT` | 30  | EBX = job_id, ECX = buffer addr, EDX = buffer size | bytes written (JSON, null-terminated), 0 if not completed, -2 if denied |

Config keys: `dataset` (name under `datasets/`), `epochs`, `lr`, `batch_size`,
`embed_dim`, `n_layer`, `n_head`.

### Sample programs

| Sample         | Demonstrates                                                                             |
| -------------- | ---------------------------------------------------------------------------------------- |
| `hello`        | VGA text-buffer write, LODSB/STOSW loop                                                  |
| `count`        | VGA digits, loop/counter                                                                 |
| `fib`          | Fibonacci, div/mod digit conversion                                                      |
| `sort`         | Bubble sort                                                                              |
| `rainbow`      | VGA colored banner                                                                       |
| `primes`       | Prime sieve, VGA output                                                                  |
| `calculator`   | Arithmetic, keyboard input                                                               |
| `factorial`    | Recursive multiplication                                                                 |
| `guess`        | Number-guessing with keyboard + VGA prompt                                               |
| `train`        | `SYS_TRAIN_START` (EAX=28) launching a Shakespeare training job                          |
| `train-status` | `SYS_TRAIN_STATUS` (29) then `SYS_TRAIN_GET_RESULT` (30), results stored to guest memory |

`train` and `train-status` require the `admin` role. After a successful `train` run the
console's Training card polls `/vm/training/jobs/{id}` until the job reaches a terminal
state, renders the returned result JSON inline, and exposes a Stop button while the job
is running (POSTs to `/vm/training/jobs/{id}/stop`). When a run executes
`SYS_TRAIN_GET_RESULT`, the returned JSON is also surfaced in the `/vm/run` response as
`training_result` and rendered in a "Training result" card.

The **Training launch** card builds training click-and-done: it edits a config
(dataset, epochs, learning rate, batch size, layers, heads, embed size), generates the
`train` assembly source with that config embedded in the `config: db '...'` literal, and
"Launch training" writes that source to the editor and runs it with the currently
selected role. "Load sample" only writes the generated source without running it. The
config fields map directly to the `TrainingRequest`/`TrainRequest` JSON keys.

The dataset field is a dropdown populated from `GET /datasets` when the backend is
reachable (a `Custom…` option reveals a free-text input); it falls back to a plain text
input when the dataset list is unavailable. Numeric fields are clamped on generate
(min 1 for integer fields, positive for learning rate, defaults restored for empty
values), so clearing a field can never emit an invalid `TrainingRequest`. The config
itself is persisted to `localStorage` (`vm-train-config`), restored on mount with the
same clamping, and a "Reset config" action restores the defaults (and clears a custom
dataset). Empty numeric fields store `NaN` and fall back to the defaults at launch;
the card lists inline "using default N" hints for any field that would fall back.
A successful launch (EAX ≥ 1 from `SYS_TRAIN_START`) shows an inline, dismissible
"Launched training job #N" confirmation in the card; a denied launch shows none
(the permission warning covers that case).
Typing a custom dataset name that is not in the loaded dataset list shows a
destructive warning ("Unknown dataset — Training will fail to start") listing up
to five available datasets, mirroring the backend's `resolve_training_inputs()`
legacy-path check (`datasets/<name>/input.txt` must exist).
When the role is `user` (training is gated behind `ADMIN`), the card shows an
inline warning with a "Switch to admin" action; `admin` and `kernel` roles can launch
directly.

## Persistence

The console persists state to `localStorage`: the assembly `source` (`vm-source`), the
selected role (`vm-role`), the steps value (`vm-max-steps`), and the training launch
config (`vm-train-config`) all survive reloads and are restored on mount (role is
validated to `user`/`admin`/`kernel`, steps and config numerics are clamped, the config
is re-clamped on restore). The steps field is clamped at run time too, so a cleared or
oversized value resolves to a valid `max_steps` in the `/vm/run` request. A `#code=`
base64 URL hash takes precedence over the saved source on load.

## Relevant Files

| File                                                   | Purpose                                                                 |
| ------------------------------------------------------ | ----------------------------------------------------------------------- |
| `packages/core-py/domains/shell/vm.py`                 | `X86VirtualSystem`, `X86Assembler`, syscall handlers, `_perm_map`       |
| `packages/core-py/domains/shell/vm_permissions.py`     | `Role`, `Permission`, `X86RBAC`                                         |
| `packages/core-py/domains/shell/vm_training_bridge.py` | `VMTrainingBridge`, `get_bridge()`                                      |
| `apps/api/server/routers/vm.py`                        | `/vm/*` endpoints, role mapping, console sessions (SSE pump, caps, TTL) |
| `apps/api/server/vm_builtins.py`                       | Shell REPL kernel source (`ls`/`cat`/`cp`/`write`/`train*`/…)           |
| `apps/web/hooks/useVmConsole.ts`                       | Session lifecycle hook: SSE subscribe, input, errors                    |
| `apps/web/components/shell/VmSessionPanel.tsx`         | Console panel UI: history, sanitize, restart, error state               |
| `apps/web/app/(app)/vm/page.tsx`                       | Console UI: role selector, samples, Training card, permission banner    |
| `apps/web/lib/vm-controller.ts`                        | `vmController.run/builtins/info/trainingJob`                            |
| `packages/core-py/tests/test_vm_rbac.py`               | RBAC enforcement tests (USER denial, ADMIN allow, full syscall lane)    |
| `apps/api/server/tests/test_vm_console_session.py`     | Session/REPL tests (boot, commands, input sanitizing, SSE lifecycle)    |
| `scripts/benchmark_vm_shell.py`                        | Shell REPL boot/latency/throughput benchmark                            |
