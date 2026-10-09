# ProcessGuard Architecture

## Context: What is PGQ?

PGQ (pugqeep) is a **general-purpose execution engine** — not model-specific. It provides `Tree` (thread pool), `GuardTree` (subprocess isolation), and `Process` abstractions for running work outside of asyncio. The goal is to replace raw `asyncio` and `multiprocessing` with a library we can tailor to our needs.

ProcessGuard is one consumer of PGQ. The autoload path runs inside a PGQ `ThreadPoolExecutor` thread, which is why subprocess-based ProcessGuard was failing — `mp.Queue` handshake between a PGQ thread and a spawned subprocess is unreliable.

## Problem (Fixed)

```
_try_lazy_guard_autoload (runs in PGQ ThreadPoolExecutor thread)
  ├── SloNetChatProvider.from_slnc() → loads model in current process ✓
  └── ProcessGuard(SUBPROCESS) → spawns subprocess via mp.Process
       └── mp.Queue handshake between PGQ thread and child → TIMEOUT
            └── Fallback to eager load → model loaded TWICE
```

## Solution

```
Autoload (startup.py)                     Manual API (controllers/models.py)
├── from_slnc() loads model               ├── ProcessGuard(SUBPROCESS)
├── ProcessGuard(THREAD, provider=...)    │   ├── ModelWorkerProcess(mp.Process)
│   └── _ThreadWorker wraps provider      │   └── Full OS isolation
│       └── No double load                └── User-triggered, separate process
└── Guard monitors in-process model
```

## Architecture

### Core Principle

**One config object, two execution strategies, clean separation.**

```
ModelConfig (dataclass)          ExecutionMode (enum)
├── slnc_path                    ├── SUBPROCESS  (manual API load)
├── model_id                     └── THREAD      (autoload / PGQ)
├── quantize
├── quant_bits              ┌─────────────────────────────┐
├── quant_mode              │        ProcessGuard          │
├── quant_clip              │  (guard logic: restart,      │
└── hf_model_cls_path       │   health, callbacks)         │
                            └──────────┬──────────────────┘
                                       │
                            ┌──────────┴──────────────────┐
                            │                              │
                   ┌────────┴────────┐           ┌────────┴────────┐
                   │ ModelWorker     │           │  _ThreadWorker   │
                   │ (mp.Process +   │           │ (threading.Thread│
                   │  Queue IPC)     │           │  + direct call)  │
                   └─────────────────┘           └──────────────────┘
```

### ModelConfig Dataclass

Eliminates parameter duplication. Both ProcessGuard and workers receive one config object.

```python
@dataclass(frozen=True)
class ModelConfig:
    """Immutable model configuration shared across guard + workers."""
    slnc_path: str | None = None
    model_id: str = "default"
    quantize: bool = False
    quant_bits: int = 8
    quant_mode: str = "symmetric"
    quant_clip: float = 0.999
    hf_model_cls_path: str | None = None
    hf_model_kwargs: dict = field(default_factory=dict)
```

### ExecutionMode Enum

```python
class ExecutionMode(Enum):
    SUBPROCESS = "subprocess"  # Manual API load — full OS isolation
    THREAD = "thread"          # Autoload / PGQ — no IPC overhead
```

### _ThreadWorker (in-process, for PGQ context)

- Loads model in a `threading.Thread` via `from_slnc()`
- Uses `threading.Queue` (not `mp.Queue`) — works reliably from PGQ threads
- Accepts optional `provider` param to skip loading (wraps pre-loaded model)
- No subprocess to crash — guard monitors thread health

### ModelWorkerProcess (subprocess, for manual API load)

- Runs model in `mp.Process` via Queue IPC
- Full OS isolation — OOM kills don't affect parent
- Used when user triggers `POST /models/load`

### ProcessGuard (simplified)

- Receives `ModelConfig` + `ExecutionMode`
- Creates appropriate worker (_ThreadWorker or ModelWorkerProcess)
- Owns: restart logic, health monitoring, callbacks, semaphore
- Delegates: generate, stream, adapter loading to worker

```python
class ProcessGuard:
    def __init__(
        self,
        model_config: ModelConfig | None = None,
        mode: ExecutionMode = ExecutionMode.THREAD,
        worker_id: str = "guard",
        provider: Any = None,  # Pre-loaded model for THREAD mode
        # ... other params
    ):
```

### Call Sites

**Autoload (startup.py) — THREAD mode, pre-loaded provider:**
```python
provider = SloNetChatProvider.from_slnc(slnc_path, ...)  # Load once
process_guard = ProcessGuard(
    mode=ExecutionMode.THREAD,
    provider=provider,  # Wrap existing provider
    slnc_path=slnc_path,
    model_id=model_type,
)
process_guard.start()
```

**Manual API (controllers/models.py) — SUBPROCESS mode:**
```python
guard = ProcessGuard(
    mode=ExecutionMode.SUBPROCESS,  # Full isolation
    slnc_path=slnc_path,
    model_id=model_id,
)
guard.start()
```

## File Changes

| File | Action | Purpose |
|------|--------|---------|
| `model_config.py` | **NEW** | `ModelConfig` dataclass + `ExecutionMode` |
| `process_guard.py` | **REFACTOR** | `_ThreadWorker` + simplified `ProcessGuard` |
| `model_worker.py` | **KEPT** | Subprocess worker (unchanged) |
| `startup.py` | **UPDATED** | Autoload uses THREAD + provider injection |
| `controllers/models.py` | **UPDATED** | Manual load uses SUBPROCESS |
| Tests | **UPDATED** | Match new API |
