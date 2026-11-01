# Logging System

## Architecture

```
domain.logging.Logger (ABC)
├── CLILogger        → CLI (ANSI color, no external deps)
├── CompositeLogger  → Multi-output fan-out
├── StructuredLogger → Key-value context, child(), tagged(), log_timer, timed
└── WebLogger        → Browser frontend (console API)
```

**Convention**: All loggers use `slo.*` prefix. Example: `logging.getLogger("slo.routers.inference")`.

## Key Files

| File | Purpose |
|------|---------|
| `domain/logging/_internal/base.py` | Logger ABC, CompositeLogger, context propagation |
| `domain/logging/_internal/cli_logger.py` | Pure ANSI CLILogger, timer() context manager |
| `domain/infrastructure/_internal/structured_log.py` | StructuredLogger, child(), tagged(), log_timer, timed decorator |
| `domain/logging/_internal/console_logger.py` | ConsoleLogger for API server |
| `domain/logging/_internal/web_logger.py` | WebLogger for browser frontend |

## Usage

```python
from domain.logging import get_logger

logger = get_logger("slo.routers.inference")
logger.info("Model loaded", extra={"tag": "MODEL", "context": {"model": "sloughgpt-7b"}})
logger.warning("High latency: %dms", elapsed_ms, extra={"tag": "PERF"})
```

## Structured Logging

```python
from domain.infrastructure._internal.structured_log import get_structured_logger, tagged, log_timer

logger = get_structured_logger("slo.training")

# Child logger with persistent context
child = logger.child(session_id="abc123")
child.info("Training started", extra={"context": {"epochs": 10}})

# Tagged for filtering
tagged(logger, "MODEL").info("Model loaded")

# Performance timing
with log_timer(logger, "inference"):
    result = model.generate(prompt)
```

## Migration Status

All diagnostic `print()` calls in core-py domains have been converted to `logger.info/warning/debug`. The 64 remaining `print()` calls are exclusively user-facing CLI output (tokenizer show methods, status summaries, demo output).
