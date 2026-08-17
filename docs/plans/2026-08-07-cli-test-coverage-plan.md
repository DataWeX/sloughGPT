# CLI Test Coverage Plan

**Date:** 2026-08-07
**Status:** Ready to implement
**Goal:** Add missing tests for untested CLI modules, bringing coverage from 26.3% to ~80%

---

## Current State

| Category | Tested | Total | Coverage |
|----------|--------|-------|----------|
| Commands | 3 (`memory`, `train`, `token_tree`) | 8 | 37.5% |
| Core | 1 (`validator`) | 6 | 16.7% |
| Utils | 1 (`training_progress`) | 4 | 25.0% |
| Entry point | 0 | 1 | 0% |
| **Overall** | **5 files** | **19 files** | **26.3%** |

**Existing tests:** 99 tests across 6 files

---

## Test Patterns to Follow

Based on existing test files, all new tests must:

1. **Path setup**: `sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))`
2. **Mock pattern**: Use `MagicMock` for args objects (not `SimpleNamespace`)
3. **Fixture pattern**: `autouse` fixtures for service mocking (see `test_memory_commands.py`)
4. **Import pattern**: Lazy imports inside test methods (not module-level)
5. **Assertion pattern**: Check both output (`capsys`) and mock calls
6. **Error pattern**: `pytest.raises(SystemExit)` for validation failures

---

## Priority 1: Pure Functions (Easy Wins)

### `utils/formatting.py` → `tests/test_formatting.py`

**Why first:** Zero dependencies, pure functions, ~15 tests, instant gratification.

| Function | Test Cases |
|----------|-----------|
| `format_size(0)` | Returns "0 B" |
| `format_size(1023)` | Returns "1023 B" |
| `format_size(1024)` | Returns "1.0 KB" |
| `format_size(1048576)` | Returns "1.0 MB" |
| `format_size(1073741824)` | Returns "1.0 GB" |
| `format_size(1536)` | Returns "1.5 KB" |
| `format_time(0)` | Returns "0s" |
| `format_time(65)` | Returns "1m 5s" |
| `format_time(3661)` | Returns "1h 1m 1s" |
| `format_number(1234567)` | Returns "1,234,567" |
| `format_number(0)` | Returns "0" |
| `truncate("hello world", 5)` | Returns "hello..." |
| `truncate("hi", 10)` | Returns "hi" (no truncation) |
| `wrap_text("a b c", 10)` | Wraps correctly |
| `indent("line1\nline2", "  ")` | Indents all lines |

### `core/version.py` → `tests/test_version.py`

| Function | Test Cases |
|----------|-----------|
| `VersionInfo.__str__` | Returns formatted string |
| `VersionInfo.to_dict` | Returns dict with keys |
| `VersionInfo.to_json` | Returns valid JSON |
| `format_version_display` | Returns colored output |

### `utils/progress.py` → `tests/test_progress.py`

| Class | Test Cases |
|-------|-----------|
| `ProgressBar` | Init, update, finish, render |
| `Spinner` | Init, next, finish |
| `progress_iter` | Yields items, calls callback |

---

## Priority 2: Core Infrastructure

### `core/cli_group.py` → `tests/test_cli_group.py`

| Feature | Test Cases |
|---------|-----------|
| `SmartGroup` fuzzy match | Exact match works |
| `SmartGroup` fuzzy match | "gpt" suggests "models" |
| `SmartGroup` fuzzy match | No match shows error |
| `SmartGroup` help format | Groups commands by category |
| `SmartGroup` error display | Shows suggestions on typo |

### `core/permissions.py` → `tests/test_permissions.py`

| Feature | Test Cases |
|---------|-----------|
| `PermissionsManager.confirm_download` | Auto-approves < 50MB |
| `PermissionsManager.confirm_download` | Prompts for > 50MB |
| `PermissionsManager.confirm_download` | User confirms → True |
| `PermissionsManager.confirm_download` | User denies → False |
| `PermissionsManager._is_cached` | Returns True if cached |
| `PermissionsManager._is_cached` | Returns False if not cached |
| `ModelSizeEstimate` | Properties work correctly |

### `core/completion.py` → `tests/test_completion.py`

| Feature | Test Cases |
|---------|-----------|
| `CompletionCache` | Cache hit returns stored value |
| `CompletionCache` | TTL expiry triggers refresh |
| `CompletionCache` | Stale-on-error uses cached |
| `CompletionCache.invalidate` | Clears cache |
| `complete_paths` | Returns matching paths |
| `COMMAND_COMPLETERS` | All keys are valid commands |

---

## Priority 3: Command Modules

### `commands/models.py` → `tests/test_models_commands.py`

| Command | Test Cases |
|---------|-----------|
| `cmd_models` | Lists models from API |
| `cmd_models` | Handles empty list |
| `cmd_models` | Handles API error |
| `_cmd_models_info` | Shows model details |
| `_cmd_models_info` | Handles missing model |
| `_cmd_models_download` | Download with progress |
| `_cmd_models_download` | User cancels |
| `_cmd_models_status` | Shows loaded model |
| `_cmd_models_status` | Shows no model |
| `_cmd_models_compare` | Compares two models |
| `_cmd_models_personalities` | Lists souls |
| `cmd_soul` | Shows current soul |
| `cmd_soul` | Switches soul |
| `cmd_export_cli` | Exports model |
| `cmd_benchmark` | Runs benchmark |

### `commands/chat.py` → `tests/test_chat_commands.py`

| Command | Test Cases |
|---------|-----------|
| `cmd_generate` | Generates text |
| `cmd_generate` | Handles empty prompt |
| `cmd_generate` | Handles API error |
| `cmd_chat` | Starts chat loop |
| `cmd_chat` | Exits on "quit" |
| `cmd_chat` | Exits on Ctrl+C |

### `commands/system.py` → `tests/test_system_commands.py`

| Command | Test Cases |
|---------|-----------|
| `cmd_status` | Shows system status |
| `cmd_status` | Watch mode updates |
| `cmd_system` | Shows system info |
| `cmd_config_check` | Validates config |
| `cmd_config_validate` | Shows validation results |
| `cmd_config_generate` | Generates secrets |
| `cmd_setup` | Runs setup |
| `cmd_stats` | Shows statistics |

### `commands/data.py` → `tests/test_data_commands.py`

| Command | Test Cases |
|---------|-----------|
| `cmd_datasets` | Lists datasets |
| `cmd_datasets` | Handles empty list |
| `cmd_dataset_stats` | Shows dataset stats |
| `cmd_dataset_search` | Searches dataset |
| `cmd_dataset_import` | Imports from GitHub |
| `cmd_dataset_import` | Imports from URL |
| `cmd_dataset_export` | Exports dataset |
| `cmd_data_tool` | Validates data |

### `commands/dev.py` → `tests/test_dev_commands.py`

| Command | Test Cases |
|---------|-----------|
| `cmd_health` | Shows health status |
| `cmd_health` | Handles server down |
| `cmd_api_status` | Shows API status |
| `cmd_api_test` | Tests API endpoint |
| `cmd_api_auth` | Tests authentication |
| `cmd_serve` | Starts server |
| `cmd_serve` | Port in use |

---

## Priority 4: Inline Commands

### `cli.py` inline commands → `tests/test_inline_commands.py`

| Command Group | Test Cases |
|---------------|-----------|
| `knowledge search` | Searches knowledge base |
| `knowledge dedup` | Deduplicates facts |
| `knowledge categorize` | Categorizes facts |
| `knowledge gaps` | Shows knowledge gaps |
| `knowledge ingest` | Ingests documents |
| `checkpoint list` | Lists checkpoints |
| `checkpoint load` | Loads checkpoint |
| `checkpoint delete` | Deletes checkpoint |
| `docker up` | Starts containers |
| `docker down` | Stops containers |
| `docker status` | Shows status |

---

## Implementation Order

1. **Week 1:** Priority 1 (pure functions) - 3 files, ~30 tests
2. **Week 1:** Priority 2 (core infrastructure) - 3 files, ~25 tests
3. **Week 2:** Priority 3 (commands) - 5 files, ~50 tests
4. **Week 2:** Priority 4 (inline commands) - 1 file, ~15 tests
5. **Week 2:** Add `apps/cli/tests/` to `pytest.ini` `testpaths`
6. **Week 2:** Create `apps/cli/conftest.py` with shared fixtures

---

## Shared Fixtures (`conftest.py`)

```python
import sys
import os
import pytest
from unittest.mock import MagicMock

# Add CLI src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def fake_args():
    """Factory for creating mock CLI argument namespaces."""
    def _factory(**kwargs):
        return MagicMock(**kwargs)
    return _factory


@pytest.fixture
def mock_api_server(monkeypatch):
    """Mock requests.get/post for API-dependent commands."""
    import requests
    mock_get = MagicMock(return_value=MagicMock(
        status_code=200,
        json=lambda: {"status": "ok"}
    ))
    mock_post = MagicMock(return_value=MagicMock(
        status_code=200,
        json=lambda: {"status": "ok"}
    ))
    monkeypatch.setattr(requests, "get", mock_get)
    monkeypatch.setattr(requests, "post", mock_post)
    return mock_get, mock_post


@pytest.fixture
def fake_logger(monkeypatch):
    """Capture log output for assertion."""
    logs = []
    import core.version as ver
    # Patch logging to capture output
    return logs
```

---

## Verification

After implementation, run:

```bash
# Run all CLI tests
cd apps/cli && python -m pytest tests/ -x -q

# Run with coverage
cd apps/cli && python -m pytest tests/ --cov=src --cov-report=term-missing

# Run from root (after adding to pytest.ini)
make test-py ARGS="apps/cli/tests/ -x -q"
```

**Target:** 80%+ coverage on all CLI modules, 150+ total tests.
