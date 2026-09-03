# Project Rules

## Agent Behavior

### SOP — Work Workflow (follow in order for every task)

1. **Create a todo on kanban** — log the task before starting work.
2. **Do the production workflow** — follow the established dev flow (branch, lint, typecheck, test).
3. **Ask leading questions** — clarify requirements before writing code.
4. **Do work in one shot** — make all edits in a single pass, no back-and-forth.
5. **Check if it's done** — verify the output matches what was asked.
6. **Clarify** — confirm with user if anything is ambiguous.
7. **Ask for more info** — if blocked or unclear, ask before guessing.
8. **Test for bugs or errors** — run lint, typecheck, and tests.
9. **Submit with a console summary** — commit with a concise summary of what changed.

### Core Rules

- **ALWAYS check if something already exists before building it.** Before creating new files, modules, or features, search the codebase for existing implementations. Use `grep`, `glob`, and `find` to check for existing code, patterns, or similar functionality. Duplicate work wastes time and creates confusion. **This has cost us hours of wasted effort — never skip this step.**
- **Do NOT make changes or delete files without explicit user approval first.** Always describe what you plan to do, wait for confirmation, then execute. Even if the user asks you to "build X" or "fix Y", confirm the approach before writing code.
- **Always use the project venv.** Check for `.venv/`, `venv/`, or `poetry env` before running Python commands. Never use bare `python` or `pip` without activating the project environment first.

## File Safety

- **NEVER delete user data files (notes, configs, data stores) without explicit user approval first.** Ask before removing any file that contains user-created content. Propose the change, explain consequences, and wait for confirmation.
- When consolidating or migrating data, always preserve the original as a backup before modifying.
- If a file is not git-tracked, treat it as irreplaceable — ask twice.

## Architecture

- **Notes** (`~/.config/dev-notes/*.md`) are the user's journal — source of truth for task metadata (sprint, gh, status, body).
- **Board** (`.kanban/board.jsonl`) is the kanban view derived from notes via sync.
- Sync is bidirectional: note status ↔ card column.

## Core Infrastructure Sync Rule

**http-client.ts is the single source of truth for all API communication.**

When backend endpoints, response envelopes, or error formats change:

1. **Update http-client.ts first** — match the new backend contract (endpoints, response shape, error codes).
2. **Propagate to consumers** — update hooks, controllers, and components that use http-client.
3. **Never bypass http-client with raw `fetch()`** — all API calls must go through `apiGet`, `apiPost`, `apiPut`, `apiDelete`, `apiPatch`, or the `request()` function.

This ensures consistent error handling, retries, caching, circuit breaking, and interceptors across the entire frontend. Bypassing http-client breaks these guarantees and creates silent bugs.
