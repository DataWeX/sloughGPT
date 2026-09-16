# ProcessGuard Architecture

## Current State (Problems)

```
ProcessGuard (535 lines, 20 params)
├── __init__ mixes: guard config + slo config + hf config
├── generate/generate_stream → delegates to ModelWorkerProcess
├── load_adapter/unload_adapter → delegates to ModelWorkerProcess (DUPLICATE)
├── _launch_worker → creates ModelWorkerProcess
├── _monitor_loop → health thread
├── _recover_from_stall → stall recovery ( scattered)
├── health → psutil import inside method
└── _memory_mb → psutil import inside method

ModelWorkerProcess (1306 lines, 12 params)
├── __init__ duplicates: slnc_path, model_id, quantize, etc.
├── start/stop → subprocess lifecycle
├── generate/generate_stream → queue-based RPC
├── health_check → heartbeat draining (3 copies)
├── load_adapter/unload_adapter → queue-based RPC
└── _drain_heartbeats → heartbeat processing
```

### Issues

| Issue | Impact |
|-------|--------|
| 20-param `__init__` | Impossible to know which params matter where |
| Duplicated config | Changes require editing 2 classes + 4 call sites |
| Adapter loading in both | ProcessGuard delegates, but also checks `alive` — logic split |
| Stall detection scattered | `_recover_from_stall`, `_drain_heartbeats`, `_STALL_TIMEOUT_S` |
| Subprocess-only | Breaks when called from PGQ threads (mp.Queue IPC fails) |
| `psutil` imported inside methods | ~1ms overhead per call |

## Proposed Architecture

### Core Principle

**One config object, two execution strategies, clean separation.**

```
ModelConfig (dataclass)          ProcessGuardMode (enum)
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
                   │ SubprocessWorker│           │   ThreadWorker   │
                   │ (mp.Process +   │           │ (threading.Thread│
                   │  Queue IPC)     │           │  + direct call)  │
                   └─────────────────┘           └──────────────────┘
```

### 1. ModelConfig Dataclass

Eliminates parameter duplication. Both ProcessGuard and workers receive one config object.

```python
@dataclass(frozen=True)
class ModelConfig:
    """Immutable model configuration shared across guard + workers."""
    __slots__ = ('slnc_path', 'model_id', 'quantize', 'quant_bits',
                 'quant_mode', 'quant_clip', 'hf_model_cls_path',
                 'hf_model_kwargs')

    slnc_path: str | None = None
    model_id: str = "default"
    quantize: bool = False
    quant_bits: int = 8
    quant_mode: str = "symmetric"
    quant_clip: float = 0.999
    hf_model_cls_path: str | None = None
    hf_model_kwargs: dict = field(default_factory=dict)

    @property
    def backend(self) -> str:
        return "slo" if self.slnc_path else "hf"

    @classmethod
    def from_env(cls, model_id: str, cfg: ServerConfig) -> "ModelConfig":
        """Build from ServerConfig + model id."""
        ...
```

### 2. ProcessGuardMode Enum

```python
class ProcessGuardMode(Enum):
    SUBPROCESS = "subprocess"  # Manual API load — full isolation
    THREAD = "thread"          # Autoload / PGQ — no IPC overhead
```

### 3. Worker Protocol (ABC)

Both execution strategies implement the same interface:

```python
class WorkerProtocol(ABC):
    """Common interface for subprocess and thread workers."""

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self, timeout: float = 10.0) -> None: ...

    @abstractmethod
    def alive(self) -> bool: ...

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> dict: ...

    @abstractmethod
    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, dict]: ...

    @abstractmethod
    def health_check(self) -> WorkerHealth: ...

    @abstractmethod
    def load_adapter(self, path: str, merge: bool = False) -> dict: ...

    @abstractmethod
    def unload_adapter(self) -> dict: ...
```

### 4. SubprocessWorker (current ModelWorkerProcess, cleaned up)

- Runs model in `mp.Process` via Queue IPC
- Used for manual API load (full isolation)
- `startup_timeout` for ready handshake
- `stall_timeout` for hung workers
- Handles generate, stream, adapter loading via Queue

### 5. ThreadWorker (new)

- Loads model directly in calling thread via `SloNetChatProvider.from_slnc()`
- No IPC — direct function calls
- Used for autoload / PGQ context
- No subprocess to crash — guard monitors thread health
- Handles generate, stream, adapter loading directly

### 6. ProcessGuard (simplified)

- Receives `ModelConfig` + `ProcessGuardMode`
- Creates appropriate worker (SubprocessWorker or ThreadWorker)
- Owns: restart logic, health monitoring, callbacks, semaphore
- Delegates: generate, stream, adapter loading to worker

```python
class ProcessGuard:
    def __init__(
        self,
        model_config: ModelConfig,
        mode: ProcessGuardMode = ProcessGuardMode.SUBPROCESS,
        worker_id: str = "guard",
        max_restarts: int = 3,
        restart_delay: float = 2.0,
        generate_timeout: float = 120.0,
        stall_timeout: float = 30.0,
        memory_limit_mb: float | None = 4096.0,
        max_concurrent: int | None = None,
    ):
        self._config = model_config
        self._mode = mode
        self._worker = self._create_worker()
        ...

    def _create_worker(self) -> WorkerProtocol:
        if self._mode == ProcessGuardMode.THREAD:
            return ThreadWorker(self._config, ...)
        return SubprocessWorker(self._config, ...)
```

### 7. Call Site Changes

**Before (startup.py — autoload):**
```python
# BROKEN — subprocess IPC from PGQ thread
process_guard = ProcessGuard(
    slnc_path=slnc_path,
    model_id=model_type,
    worker_id=f"slo-{model_type.split('/')[-1]}",
    max_restarts=3,
    restart_delay=2.0,
    generate_timeout=cfg.generate_timeout,
    memory_limit_mb=resolve_memory_limit_mb(slnc_path, cfg.process_guard_memory_limit_mb),
    quantize=cfg.quantize_slonet,
    quant_bits=cfg.quant_bits,
    quant_mode=cfg.quant_mode,
    quant_clip=cfg.quant_clip,
)
process_guard.start()
```

**After (startup.py — autoload):**
```python
config = ModelConfig.from_env(model_type, cfg)
process_guard = ProcessGuard(
    model_config=config,
    mode=ProcessGuardMode.THREAD,
    worker_id=f"slo-{model_type.split('/')[-1]}",
)
process_guard.start()
```

**Before (controllers/models.py — manual load):**
```python
guard = ProcessGuard(
    slnc_path=slnc_path,
    model_id=model_id,
    worker_id=f"slo-{model_id.split('/')[-1]}",
    max_restarts=3,
    restart_delay=2.0,
    generate_timeout=cfg.generate_timeout,
    memory_limit_mb=resolve_memory_limit_mb(slnc_path, cfg.process_guard_memory_limit_mb),
    quantize=cfg.quantize_slonet,
    quant_bits=cfg.quant_bits,
    quant_mode=cfg.quant_mode,
    quant_clip=cfg.quant_clip,
)
guard.start()
```

**After (controllers/models.py — manual load):**
```python
config = ModelConfig.from_env(model_id, cfg)
guard = ProcessGuard(
    model_config=config,
    mode=ProcessGuardMode.SUBPROCESS,
    worker_id=f"slo-{model_id.split('/')[-1]}",
)
guard.start()
```

## File Changes

| File | Action | Lines |
|------|--------|-------|
| `model_config.py` | **NEW** — ModelConfig dataclass | ~50 |
| `worker_protocol.py` | **NEW** — WorkerProtocol ABC | ~40 |
| `thread_worker.py` | **NEW** — ThreadWorker impl | ~200 |
| `subprocess_worker.py` | **RENAME** from model_worker.py, cleaned up | ~900 |
| `process_guard.py` | **REFACTOR** — simplified, uses WorkerProtocol | ~300 |
| `startup.py` | Update call sites | ~10 lines |
| `controllers/models.py` | Update call sites | ~15 lines |
| Tests | Update + new tests | ~200 lines |

## Migration Path

1. Create `ModelConfig` dataclass — no breaking changes
2. Create `WorkerProtocol` ABC — no breaking changes
3. Create `ThreadWorker` — new code, no existing behavior affected
4. Rename `model_worker.py` → `subprocess_worker.py` + update imports
5. Refactor `ProcessGuard` to use `WorkerProtocol`
6. Update call sites to use `ModelConfig`
7. Run full test suite
