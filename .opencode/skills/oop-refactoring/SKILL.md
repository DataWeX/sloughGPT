---
name: oop-refactoring
description: >
  Use when refactoring Python or TypeScript code to improve OOP structure,
  reduce memory usage, extract classes, add __slots__, create metaclasses,
  or fix code smells. Works with QA to verify no regressions.
---

# OOP Refactoring Skill

## Overview

This skill provides systematic guidance for refactoring code using OOP principles
to improve memory efficiency, reduce CPU overhead, and enhance maintainability.

## When to Use

- Code has duplicated initialization patterns across classes
- Classes are larger than 300 lines (god class anti-pattern)
- Hot path code imports modules inside functions
- Properties have side effects
- Singleton patterns are repeated 5+ times
- Dataclasses don't use `__slots__`
- Memory profiling shows excessive allocation

## Refactoring Checklist

### Pre-Refactoring
- [ ] Read existing tests to understand expected behavior
- [ ] Run `python3 -m py_compile` on files to be modified
- [ ] Run relevant test suite to establish baseline
- [ ] Document current memory/CPU usage (if available)

### During Refactoring
- [ ] Make ONE change at a time
- [ ] Create new files before modifying existing ones
- [ ] Update imports in all dependent files
- [ ] Run `python3 -m py_compile` after each file change
- [ ] Run relevant tests after each logical change

### Post-Refactoring
- [ ] Run full test suite
- [ ] Run type checkers (`npx tsc --noEmit`, `mypy`)
- [ ] Verify no import errors
- [ ] Document expected memory/CPU savings
- [ ] Get QA sign-off

## Common Patterns

### 1. Extract Session Manager
```python
# When: Multiple classes manage session state identically
# Before: 5 attributes duplicated across 3 construction paths
instance._kv_states = {}
instance._kv_last_access = {}
instance._kv_ttl = 3600.0
instance._kv_max_sessions = 64
instance._kv_lock = threading.Lock()

# After: Single manager class
@dataclass
class SessionKVManager:
    __slots__ = ('kv_states', 'kv_last_access', 'kv_ttl', 'kv_max_sessions', 'lock')
    kv_states: Dict[str, Any] = field(default_factory=dict)
    kv_last_access: Dict[str, float] = field(default_factory=dict)
    kv_ttl: float = 3600.0
    kv_max_sessions: int = 64
    lock: threading.Lock = field(default_factory=threading.Lock)
```

### 2. Add `__slots__` to Dataclasses
```python
# When: Dataclass instances are created frequently
# Memory savings: ~300 bytes → ~80 bytes per instance

@dataclass
class Task:
    __slots__ = ('id', 'name', 'status', 'result')
    id: str
    name: str
    status: TaskStatus
    result: Any = None
```

### 3. Singleton Metaclass
```python
# When: 5+ modules use the same double-checked locking pattern
# Before: 10+ copies of boilerplate
_lock = threading.Lock()
_instance = None

def get_thing():
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = Thing()
    return _instance

# After: One import
from domains.infrastructure.singleton import SingletonMeta

class Thing(metaclass=SingletonMeta):
    pass
```

### 4. Module-Level Imports
```python
# When: Hot path functions import inside the function body
# Before: ~1-2ms overhead per call
def _memory_mb(self):
    try:
        import psutil
    except ImportError:
        return None

# After: Zero overhead after first call
try:
    import psutil
    _psutil_available = True
except ImportError:
    psutil = None
    _psutil_available = False

def _memory_mb(self):
    if not _psutil_available:
        return None
```

### 5. Side-Effect-Free Properties
```python
# When: Property getter modifies state (anti-pattern)
@property
def state(self) -> State:
    with self._lock:
        if self._state == State.OPEN:
            if time.time() - self._last_failure > self.timeout:
                self._transition_to(State.HALF_OPEN)  # SIDE EFFECT!
        return self._state

# After: Pure read + explicit method
@property
def state(self) -> State:
    with self._lock:
        return self._state

def allow_request(self) -> bool:
    with self._lock:
        self._maybe_transition()  # Explicit call
        return self._state != State.OPEN
```

## Testing Protocol

### Python Changes
```bash
# Syntax check
python3 -m py_compile <file>

# Run specific tests
make test-py ARGS="tests/test_<module>.py -x -q"

# Full suite
make test-py
```

### TypeScript Changes
```bash
cd apps/web

# Type check
npm run typecheck

# Run tests
npm run test:lib      # Fastest
npm run test:changed  # Only changed files
npm run test          # Full suite
```

## Memory Impact Assessment

| Optimization | Per-Instance Savings | Scale Impact |
|--------------|---------------------|--------------|
| `__slots__` on dataclass | ~220 bytes | 1000 instances = ~220KB |
| Extract manager class | ~100 lines duplicated | Reduced code surface |
| Module-level import | ~1-2ms per call | 1000 calls/sec = ~1-2s saved |
| Singleton metaclass | ~20 lines boilerplate | 10 singletons = ~200 lines saved |

## Risk Mitigation

1. **Never skip tests** — Every change must pass existing tests
2. **One change at a time** — Easier to identify regressions
3. **Preserve API** — Public interfaces must remain unchanged
4. **Document changes** — Future developers need to understand decisions
5. **Get QA sign-off** — Human verification before merging

## File Organization

```
packages/core-py/domains/
├── infrastructure/
│   ├── singleton.py          # SingletonMeta metaclass
│   ├── server_state.py       # AtomicRef with __slots__
│   ├── task_queue.py         # Task with __slots__
│   ├── model_server.py       # CircuitBreaker with __slots__
│   └── process_guard.py      # Module-level psutil import
├── inference/
│   ├── session_kv_manager.py # Extracted KV cache manager
│   └── slonet_provider.py    # Uses SessionKVManager
```
