# Session: Serve In-Place Updates + Monitoring Fix

## What was done

### 1. Monitoring page crash fix
- **Backend** (`multimodal.py:163`): DPO fallback status now returns `accepted_count: 0, rejected_count: 0, last_run: null, result: null` — matching the initialized state shape
- **Frontend** (`FeedbackCard.tsx:54-55`): Added `?.accepted_count != null` guards — renders `<Skeleton>` if fields are missing

### 2. In-place status updates for all serve commands
Added `StatusBlock` class to `dev.py` and cursor control methods to `CLILogger`:
- `cursor_up(n)`, `cursor_down(n)`, `clear_line()`, `clear_lines(n)`, `save_position()`, `restore_position()`
- `StatusBlock.update(*lines)` — clears previous lines and redraws in-place on TTY

Applied to all 4 serve paths:
- `_cmd_api_and_web`: Shows `SloughGPT` block with API/Web status, updates in-place as services become ready
- `_cmd_api_only`: Shows `SloughGPT API` block with status
- `_cmd_api_and_mobile`: Shows `SloughGPT` block with API/Mobile status
- `cmd_dev`: Shows `SloughGPT Dev` block before TUI dashboard takes over

Removed redundant `api_log.success()`, `web_log.success()`, `web_log.step()` calls that were printing alongside the status block. Replaced `_wait_for_api_with_progress()` spinner with inline polling that updates the status block.

### 3. Files changed
- `apps/cli/src/commands/dev.py`: StatusBlock class, all 4 serve paths rewritten
- `packages/core-py/domains/logging/cli_logger.py`: Added cursor control methods
- `apps/api/server/routers/multimodal.py`: DPO fallback fields
- `apps/web/components/monitoring/FeedbackCard.tsx`: Optional chaining guards

## Status
- All files pass `py_compile`
- Serve output updates in-place on TTY (no more duplicate lines)
- Monitoring page no longer crashes when DPO is idle
