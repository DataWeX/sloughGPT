# Dait Shell

## Overview

Dait is a full-featured interactive command shell that connects to
your local AI backend. It provides 40+ built-in commands, pipeline chaining, background
execution, output redirection, environment variables, aliases, file/directory completion,
a pager, persistent state, script execution, inline Python evaluation, and LLM-powered
natural language interpretation — all in a single Python module.

```
          ┌─────────────────┐    pipelines/redirection     ┌──────────────┐
          │  ShellREPL      │ ──────────────────────────▶  │  DaitRuntime  │
          │  (repl.py)      │ ◀──────────────────────────  │  (kernel.py) │
          │  40+ commands   │    state persistence          │  process mgmt│
          │  tab completion │                               │  resource    │
          │  pager / less   │                               │  scanning    │
          └────────┬────────┘                               └──────┬───────┘
                   │ API calls (requests)                          │
                   ▼                                                ▼
          ┌─────────────────┐                               ┌──────────────┐
          │  ShellCommands  │                               │  ShellState  │
          │  (commands.py)  │                               │  (state.py)  │
          │  22 API methods │                               │  JSON-backed │
          └─────────────────┘                               └──────────────┘
```

**Stack:** Python 3.9+, `requests`, `readline` (optional), `json`.

---

## Installation

The shell is part of `packages/core-py/domains/shell/`. There are no extra dependencies
beyond Python 3.9+ and `requests` (for API calls to the backend).

### Launching

```bash
# Interactive REPL
sloughgpt shell

# Single command (non-interactive)
sloughgpt shell -c "health"
sloughgpt shell -c "models | grep gpt"
sloughgpt shell -c "gen hello world > output.txt"
```

### Programmatic Use

```python
from domains.shell.repl import ShellREPL
from domains.shell.kernel import DaitRuntime

rt = DaitRuntime()
repl = ShellREPL(rt)

# Run a single command and capture output
output = repl._execute_single("health")
print(output)

# Run a pipeline
repl._execute_pipeline(["models", "grep gpt"])

# Start interactive loop
repl.run()
```

---

## Getting Started

### First Run

On first launch, the shell shows a welcome message with quick-start commands.
Type `tutorial` for an interactive walkthrough covering health checks, model
listing, pipelines, background tasks, redirection, aliases, PS1, and AI mode.

### Command Syntax

```
<command> [arguments]
<command> | <command>          # Pipeline
<command> &                     # Background
<command> > <file>              # Redirect (overwrite)
<command> >> <file>             # Redirect (append)
time <command>                  # Time execution
NAME=VALUE <command>            # Inline env var
$(<command>)                    # Command substitution
```

### Tab Completion

- **First word:** Commands and aliases
- **Arguments:** Dynamic model/soul/dataset/checkpoint names (from API)
- **Files/paths:** Automatic for `source`, `less`, `tee`, `pushd` and any
  command without a specific completion handler

---

## Built-in Commands

### System & Health

| Command | Description |
|---------|-------------|
| `health` | API status, loaded model, active soul (colored) |
| `status` | Kernel uptime, processes, model, soul, memory |
| `metrics` | CPU, memory, disk metrics from the server |

### Model Management

| Command | Description |
|---------|-------------|
| `models` | List available models (tab-completes names) |
| `load <name>` | Load a model by name |
| `unload` | Unload the current model |
| `gen <prompt>` | Generate text with the loaded model |

### Souls & Personality

| Command | Description |
|---------|-------------|
| `souls` | List available personality profiles |
| `switch <name>` | Switch to a soul |
| `whoami` | Show current soul |

### Data

| Command | Description |
|---------|-------------|
| `datasets` | List datasets (tab-completes names) |
| `knowledge` | List knowledge base entries |
| `checkpoints` | List training checkpoints (tab-completes names) |
| `finetuned` | List fine-tuned model paths (tab-completes names) |
| `tokenizer` | Show tokenizer vocabulary stats |

### Process Management

| Command | Description |
|---------|-------------|
| `procs` / `ps` | List running training jobs |
| `kill <id>` | Stop a training job |

### History & Navigation

| Command | Description |
|---------|-------------|
| `history [n]` | Show command history (last 20, or n) |
| `fc [-l] [n]` | List history or re-run command #n |
| `pushd <dir>` | Push directory onto stack and cd |
| `popd` | Pop directory stack and cd back |
| `dirs` | Show directory stack |

### Aliases & Environment

| Command | Description |
|---------|-------------|
| `alias [name=cmd]` | List or set aliases |
| `unalias <name>` | Remove an alias |
| `set [name=value]` | Set or show environment variables |
| `export [NAME=VALUE]` | POSIX-style export (delegates to `set`) |

### Scripting & Evaluation

| Command | Description |
|---------|-------------|
| `source <file>` / `.` | Execute commands from a file |
| `py <expr>` | Evaluate a Python expression |
| `ai <query>` | Natural language → shell command |

### Pipe Filters

| Command | Description |
|---------|-------------|
| `grep <pattern>` | Filter lines by regex |
| `head [n]` | Show first n lines (default 10) |
| `tail [n]` | Show last n lines (default 10) |
| `wc` | Count lines, words, characters |
| `tee <file>` | Write piped input to file + pass through |
| `sort [-r] [-u] [-n]` | Sort lines; reverse/unique/numeric |
| `uniq` | Deduplicate consecutive lines |
| `less` | Pager: scroll page by page |
| `echo <text>` | Print text |

### Shell Control

| Command | Description |
|---------|-------------|
| `help [cmd]` | Show help or help for a command |
| `clear` / `cls` | Clear the screen |
| `exit` / `q` / `quit` | Exit the shell |
| `tutorial` | Interactive walkthrough |
| `sleep <sec>` | Pause for N seconds |
| `watch <sec> <cmd>` | Run command repeatedly every N seconds |
| `bg` / `jobs` | List background shell processes |
| `fg <id>` | Wait for a background process |

---

## Pipelines

Commands can be chained with `|`. The output of the left command becomes the
piped input (`_piped_input`) of the right command.

```bash
# Two-stage pipeline
models | grep gpt

# Three-stage
models | grep gpt | wc

# With filters
health | head 3
echo "a\nb\nc" | sort -r
models | grep llama | tee models.txt
history | grep health | less
```

Pipeline filters: `grep`, `head`, `tail`, `wc`, `tee`, `sort`, `uniq`, `less`, `echo`.

---

## Background Execution

Append `&` to run a command in a daemon thread without blocking the prompt.

```bash
health &
models | grep gpt &
gen hello world > out.txt &
```

Job control:

```bash
bg          # List background processes (alias: jobs)
fg 1        # Wait for background process #1
```

Background threads are daemon threads — they terminate when the shell exits.

---

## Output Redirection

```bash
# Overwrite
gen hello world > output.txt

# Append
models >> models.txt

# Pipeline with redirection
models | grep gpt > gpt-models.txt
```

Redirection is parsed from the end of the command line. Works with pipelines
and background commands.

---

## Environment Variables

```bash
# Persistent (saved to shell_state.json)
set MY_VAR=hello
set PS1='\u@\h \w $ '

# Show value
set MY_VAR

# Show all
set

# POSIX-style
export MY_VAR=hello

# Inline (set for one command only)
MY_VAR=world echo $MY_VAR

# Variable expansion
echo $HOME
echo ${MY_VAR:-default}

# Command substitution
echo $(whoami)
echo "Model count: $(models | wc)"
```

### Special Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PS1` | `λ` | Prompt string (see escapes below) |
| `SHELL` | `sloughgpt` | Shell name |
| `HOME` | `~` | Home directory |
| `NO_COLOR` | unset | Set to `1`/`true`/`yes` to disable ANSI colors |

### PS1 Escape Sequences

| Escape | Expands to |
|--------|------------|
| `\h` | Hostname (short) |
| `\w` | Current working directory (`~` for home) |
| `\t` | Current time (HH:MM:SS) |
| `\u` | Current user |
| `\s` | Shell name (`sloughgpt`) |
| `\#` | Command number |
| `\n` | Newline |

Example: `set PS1='\u@\h \w \$ '` → `user@host ~/project $`

---

## Aliases

```bash
# Set
alias ll=procs
alias h=health
alias gm='models | grep'

# List
alias

# Show single
alias h

# Remove
unalias ll
```

Aliases persist to `~/.config/sloughgpt/shell_state.json` across sessions.

### Default Aliases

| Alias | Expands to |
|-------|------------|
| `q` | `exit` |
| `quit` | `exit` |
| `h` | `help` |
| `?` | `help` |
| `cls` | `clear` |
| `ps` | `procs` |
| `jobs` | `bg` |

---

## History

```bash
history         # Show last 20 commands
history 5       # Show last 5 commands
fc              # List history (same as history)
fc -l 5         # List last 5
fc 42           # Re-run command #42
```

History is saved to `~/.config/sloughgpt/shell_state.json` (max 500 entries).

### History Search

| Keybinding | Action |
|------------|--------|
| `Ctrl+R` | Reverse search (backward) |
| `Ctrl+S` | Forward search |

These use readline's built-in incremental search.

### TUI (split-pane shell)

Launch with `sloughgpt tui`, `sloughgpt shell --tui`, the `tui` REPL command,
or `MAN_TUI=1 sloughgpt shell`. Line mode is the default. Three panes: console
logs (top), command output (middle), and a chrome bar + input row at the
bottom. Commands run on a background thread so output streams live.

| Keybinding | Action |
|------------|--------|
| `Tab` | Complete command or path |
| `Up` / `Down` | Command history (then arrows move caret) |
| `Left` / `Right` | Move caret |
| `Alt+F` / `Ctrl+Right` | Move forward to the end of the next word |
| `Alt+B` / `Ctrl+Left` | Move back to the start of the current or previous word |
| `Home` / `Ctrl+A` | Jump to start of line |
| `End` / `Ctrl+E` | Jump to end of line |
| `Backspace` / `Delete` / `Ctrl+D` | Delete before / at caret |
| `Ctrl+W` | Delete word before caret (added to kill ring) |
| `Alt+D` | Delete word after caret (added to kill ring) |
| `Ctrl+T` | Transpose the characters before and at the caret (last two at end of line) |
| `Ctrl+U` | Kill to start of line (added to kill ring) |
| `Ctrl+K` | Kill to end of line (added to kill ring) |
| `Ctrl+Y` | Yank the most recent kill; press again to cycle to older kills |
| `Ctrl+R` / `Ctrl+S` | Reverse / forward incremental history search (same key again — or `Ctrl+F` — moves through matches; `Esc` cancels) |
| `/` | Search the output pane; type a query to jump the scroll so the match lands at the top (`n` / `N` at an empty prompt repeat the last accepted search; `Enter` accepts, `Esc` cancels) |
| `Ctrl+L` | Clear output pane |
| `Ctrl+O` | Toggle output/log scrollback focus |
| `PgUp` / `PgDn` | Scroll focused pane (10 lines) |
| `Ctrl+C` | Interrupt the running command (press again, or `exit`, to quit) |

The input row scrolls horizontally when the command line exceeds the
terminal width; the caret always stays visible.

---

## Tab Completion

### Built-in Candidates

| Command | Source |
|---------|--------|
| `load`, `unload`, `gen` | Model names from `/models` API |
| `switch` | Soul names from `/souls` API |
| `datasets` | Dataset names from `/datasets` API |
| `checkpoints` | Checkpoint names from `/auto-train/checkpoints` API |
| `finetuned` | Fine-tuned model names from `/training/finetuned-models` API |
| `source`, `less`, `tee`, `pushd`, `sort`, `uniq` | File/directory path completion (fallback) |

### Path Completion

Any command without a specific completion handler falls back to filesystem
path completion. Directories are shown with a trailing `/`. Hidden files
(leading `.`) are excluded.

---

## Pager

The `less` command paginates long output page by page:

```bash
models | less
history | less
```

- **Enter** — Next page
- **q** — Quit
- Content shorter than terminal height is shown without paging

---

## LLM Natural Language Mode

```bash
ai show me running training jobs
ai list all available models
ai check server health
ai show datasets
```

The `ai` command sends your query to `/inference/generate` with the full
list of available commands as context. The LLM returns a single command
which the shell executes directly.

### Fallback

If the LLM is unavailable (model not loaded), the shell falls back to
keyword-based matching:

- "process", "job", "running" → `procs`
- "model", "models" → `models`
- "soul", "personality" → `whoami`
- "health", "status" → `health`
- "dataset", "data" → `datasets`
- "knowledge", "fact" → `knowledge`
- "checkpoint" → `checkpoints`
- "finetune", "trained" → `finetuned`
- "metric", "cpu", "memory", "disk" → `metrics`
- "tokenizer", "vocab" → `tokenizer`
- "help", "command" → `help`

---

## Script Execution

```bash
# Run commands from a file
source setup.sh
. setup.sh
```

The `source` command reads a file line by line. Lines starting with `#` are
skipped. Errors are reported per-line without aborting the script.

### `.sloughgptrc` Startup File

`~/.config/sloughgpt/rc` is executed automatically on shell startup (like
`.bashrc`). Use it for persistent aliases and env setup:

```bash
# ~/.config/sloughgpt/rc
alias h=health
alias gm='models | grep'
set PS1='\u@\h \w \$ '
```

---

## Inline Python

```bash
py 2 + 2
py 'hello'.upper()
py [i*i for i in range(5)]
py __import__('json').dumps({'a': 1})
```

The `py` command evaluates any Python expression using `eval()` with all
builtins available. Errors are caught and displayed.

---

## Scripting Features

### Command Substitution

```bash
echo $(whoami)
echo "Model count: $(models | wc)"
```

`$(command)` is replaced with the captured output of that command. Nested
substitution is supported.

### Inline Environment Variables

```bash
MY_VAR=world echo $MY_VAR    # prints "world"
MY_VAR=hello                  # error: no command
```

Variables set inline are available for that command only and are cleaned
up afterward. `$VAR` expansion picks up the inline value.

### Execution Timing

```bash
time health
time load gpt2
time models | grep gpt
```

Prefix any command or pipeline with `time` to measure elapsed wall-clock
time. Displayed as `[0.42s]` after output.

### Multiline Input

End a line with `\` to continue on the next line:

```bash
λ models | grep llama \
  > > models.txt
```

The secondary prompt is `  > `.

---

## Job Control

```bash
health &
bg          # List (alias: jobs)
fg 1        # Wait for background process #1
```

Background processes run in daemon threads. Their output appears
interleaved with the prompt when they complete.

---

## Keyboard Shortcuts

Editing comes from the system `readline` library (Emacs bindings). If
`readline` is unavailable, input falls back to plain lines without editing.

### Completion & History

| Shortcut | Action |
|----------|--------|
| `Tab` | Complete command / argument / path |
| `↑` / `↓` (`Ctrl+P` / `Ctrl+N`) | Previous / next history entry |
| `Ctrl+R` | Reverse incremental history search |
| `Ctrl+S` | Forward incremental history search — subject to terminal flow control; see note below |

`Ctrl+S` is bound to `forward-search-history`, but most terminals reserve it as
XOFF (pause output) flow control, so it is consumed by the terminal and never
reaches readline — the same limitation applies to bash. Use `Ctrl+R` (press
again) to move backward through matches, or disable flow control with
`stty -ixon` to make forward search reachable.

### Cursor movement

| Shortcut | Action |
|----------|--------|
| `Ctrl+A` / `Home` | Move to start of line |
| `Ctrl+E` / `End` | Move to end of line |
| `Alt+B` | Back one word |
| `Alt+F` | Forward one word |

### Editing

| Shortcut | Action |
|----------|--------|
| `Backspace` | Delete char before caret |
| `Ctrl+D` | Delete char at caret (EOF on empty line = exit) |
| `Alt+D` | Delete word after caret |
| `Ctrl+W` | Delete word before caret |
| `Ctrl+K` | Kill to end of line |
| `Ctrl+U` | Kill to start of line |
| `Ctrl+T` | Transpose chars around caret |

### Kill ring & process control

| Shortcut | Action |
|----------|--------|
| `Ctrl+Y` | Yank last killed text |
| `Ctrl+L` | Clear screen |
| `Ctrl+C` | Abort the running command, or cancel the current line |

---

## State Persistence

All state is stored in `~/.config/sloughgpt/shell_state.json`:

```json
{
  "history": ["health", "models", "load gpt2", ...],
  "aliases": { "h": "health", "ll": "procs" },
  "env": { "PS1": "\\u@\\h \\w $ ", "MY_VAR": "hello" },
  "last_session": "2026-06-02T09:00:00",
  "first_run": false
}
```

- History is capped at 500 entries
- Saved after every command and on exit
- The `first_run` flag triggers the welcome message once

---

## NO_COLOR Support

Set `NO_COLOR=1` to disable all ANSI escape codes:

```bash
set NO_COLOR=1
```

This follows the [NO_COLOR](https://no-color.org/) standard. Can also be
set in the environment before launching the shell.

---

## Architecture

### Module Layout

```
domains/shell/
├── __init__.py     # Package exports
├── kernel.py       # DaitRuntime + Kernel (process/resource management)
├── repl.py         # ShellREPL (40+ commands, pipelines, readline)
├── commands.py     # ShellCommands (22 API wrappers)
└── state.py        # ShellState (JSON-backed persistence)
```

### Execution Flow

```
User input → _parse_pipeline() → alias expansion → cmd subst → inline env
  → var expansion → redirection parsing → command dispatch
  → _CaptureOutput → redirect handling → output
```

### Pipeline Flow

```
Command A output → _piped_input → Command B → _piped_input → Command C
```

Each pipeline segment stores its captured output in `self._piped_input`,
which the next segment reads. The final segment prints to stdout.

### Threading

- Background commands (`&`) run in `threading.Thread` (daemon)
- Pipeline execution is synchronous within the main thread
- `fg` uses `thread.join(timeout=600)` to wait

---

## Testing

```bash
# Unit tests (148 tests)
cd packages/core-py
python3 -m pytest tests/test_shell_repl.py -v

# Integration tests (30 tests, requires running API server)
python3 -m pytest tests/test_shell_integration.py -v

# Both
python3 -m pytest tests/test_shell_repl.py tests/test_shell_integration.py -v
```

### Test Coverage

| Area | Tests |
|------|-------|
| Pipeline parsing | 8 |
| Pipeline execution | 6 |
| Pipe filters (grep/head/tail/wc) | 4 |
| Alias (set/list/remove/expansion) | 7 |
| State persistence (save/load/dedup/max) | 8 |
| Background parsing | 3 |
| Echo | 2 |
| Source command | 5 |
| Py command | 5 |
| Command substitution | 4 |
| Env persistence | 3 |
| Help | 3 |
| History with n | 3 |
| Fc command | 6 |
| Job control (bg/fg) | 5 |
| NO_COLOR, inline env | 8 |
| Sleep | 4 |
| PS1 escapes | 7 |
| RC file | 5 |
| Gen completion, path completion | 7 |
| Tee | 2 |
| Sort (reverse/unique/numeric) | 6 |
| Uniq (dedup) | 3 |
| Less (pager) | 5 |
| Dir stack (pushd/popd/dirs) | 6 |
| Watch | 2 |
| Export | 4 |
| Command registration | 10 |
| Integration API calls (ShellCommands) | 11 |
| Integration via REPL (commands + pipelines) | 19 |

---

## API Endpoints Used

The shell communicates with the backend via HTTP. All endpoints are
documented in `docs/routers.md`.

| Shell Command | API Endpoint | Method |
|---------------|-------------|--------|
| `health` | `/health` | GET |
| `status` | `/health` + `/health/detailed` | GET |
| `models` | `/models` | GET |
| `load` | `/models/load` | POST |
| `unload` | `/models/unload` | POST |
| `souls` | `/souls` | GET |
| `switch` | `/souls/switch` | POST |
| `whoami` | `/souls/current` | GET |
| `datasets` | `/datasets` | GET |
| `knowledge` | `/knowledge/list` + `/knowledge/stats` | GET |
| `checkpoints` | `/auto-train/checkpoints` | GET |
| `load_checkpoint` | `/auto-train/checkpoints/{name}/load` | POST |
| `delete_checkpoint` | `/auto-train/checkpoints/{name}` | DELETE |
| `finetuned` | `/training/finetuned-models` | GET |
| `load_finetuned` | `/training/finetuned-models/{name}/load` | POST |
| `delete_finetuned` | `/training/finetuned-models/{name}` | DELETE |
| `gen` | `/inference/generate` | POST |
| `chat` | `/chat` | POST |
| `procs` | `/training/jobs` | GET |
| `kill` | `/training/jobs/{id}/stop` | POST |
| `metrics` | `/system/metrics` | GET |
| `tokenizer` | `/tokenizer/stats` | GET |
| `ai` | `/inference/generate` | POST |

---

## Permissions

The shell gates destructive operations by risk level. Every command is classified, and blocked commands require explicit grant before execution.

### Risk Levels

| Level | Default | Commands | Examples |
|-------|---------|----------|----------|
| **safe** | allow | Read-only, no side effects | `ls`, `cat`, `echo`, `help`, `pwd`, `grep`, `wc` |
| **elevated** | allow | Modifies shell state | `alias`, `cd`, `set`, `export`, `py`, `ai`, `bg`, `fg` |
| **dangerous** | **deny** | Modifies filesystem | `rm`, `cp`, `mv`, `chmod`, `mkdir`, `touch` |
| **critical** | **deny** | Affects system/services | `boot`, `shutdown`, `svc`, `load`, `train`, `kill` |

Force patterns auto-promote risk: `rm -rf` → critical (even though `rm` is dangerous).

### Commands

```
permit <cmd>                 Grant permission for this session
permit <cmd> --persist       Grant and save to ~/.config/shell_permissions.json
permit --all-dangerous       Allow all dangerous commands at once
deny <cmd>                   Revoke a previously granted permission
permissions                  Show current policy and granted commands
```

### Interactive Flow

When a blocked command is run, the shell prompts:

```
⚡ rm requires dangerous permissions.
Allow this? [y/N/always]
```

- **y** → grant for this session
- **always** → grant persistently (saved to disk)
- **N** or empty → deny, command skipped (exit code 126)

In programmatic mode (`execute()`), blocked commands are silently denied with exit code 126.

### Architecture

```
User input
  → run() / execute()
    → _check_permission(cmd, args)
      → ShellPermissions.check(cmd, args)
        → classify(cmd) → risk level
        → policy[risk] → allow/deny
        → granted set → override?
      → PermissionError → interactive prompt or silent deny
    → handler(self, args)
    → audit log
```

All paths covered:
- `run()` main loop → `_check_permission(interactive=True)`
- `execute()` programmatic API → `_check_permission(interactive=False)`
- `_execute_single()` pipelines/background/fc → `_check_permission(interactive=False)`

### Persistence

Grants persist to `~/.config/shell_permissions.json`:

```json
{
  "granted": ["rm", "chmod"],
  "policy": {
    "safe": "allow",
    "elevated": "allow",
    "dangerous": "deny",
    "critical": "deny"
  }
}
```

### Configuration

```python
from domains.shell import ShellPermissions, Risk

perms = ShellPermissions()
perms.set_policy(Risk.DANGEROUS, "allow")   # allow all dangerous
perms.set_policy(Risk.CRITICAL, "deny")     # keep critical blocked
perms.grant("rm", persist=True)             # allow rm forever
perms.revoke("rm")                          # remove rm grant
```

---

## CLI Integration

The shell is exposed via Click in `apps/cli/src/cli.py`:

```bash
sloughgpt shell                    # Interactive REPL
sloughgpt shell -c "health"       # Single command
sloughgpt shell --command 'models | grep gpt | wc'
```

The `--command` flag routes through `_parse_pipeline` and `_execute_single`,
so pipelines, redirection, background, and timing all work in one-shot mode.

---

## Hacking

### Adding a New Command

1. Add a `_cmd_<name>` method to `ShellREPL` in `repl.py`
2. Register it in the `COMMANDS` dict
3. Add a help entry in `_cmd_help`
4. Optionally add tab completion in `_complete_args_for`
5. Write tests in `tests/test_shell_repl.py`
6. Add API method to `ShellCommands` if needed

### Adding a New API Endpoint

1. Add a `@staticmethod` to `ShellCommands` in `commands.py`
2. Use `_api_get`, `_api_post`, or `_api_delete` helpers
3. Wire the command handler in `repl.py`

### Adding a New Pipe Filter

1. Create `_cmd_<filter>` that reads `self._piped_input`
2. Register in `COMMANDS` dict
3. Add to help text under **Pipe filters**
