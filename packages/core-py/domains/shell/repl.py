"""
ShellREPL — interactive shell with pipelines, backgrounds, readline,
state persistence, and LLM-powered natural language interpretation.

Features:
  - 40+ built-in commands delegating to real backend endpoints
  - Pipeline chaining (|) — passes captured output as next command's args
  - Background execution (&) — spawns in a thread
  - Command chaining (&&, ||, ;) with exit code tracking ($?)
  - AI-domain commands: gen, chat, ai, models, souls, train, datasets, ...
  - Shell essentials: history, alias, export, py, jobs, watch, ...
  - readline tab completion for command names
  - Persistent history and aliases (~/.config/sloughgpt/shell_state.json)
  - LLM-powered natural language interpretation (ai <query>)
  - Alias management (alias / unalias)
"""

from __future__ import annotations

import io
import os
import re
import sys
import json
import time
import stat
import shutil
import logging
import glob
import threading
import subprocess
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .runtime import DaitRuntime
from .commands import ShellCommands
from .console import Console
from .state import ShellState

_EM = "\u2014"  # em dash
logger = logging.getLogger("slo.shell.repl")

# ── Process-wide log handlers (shared by every ShellREPL instance) ───
#
# A RotatingFileHandler holds an open file descriptor, and a LogBufferHandler
# accumulates on the shared "slo" logger. Creating one per ShellREPL instance
# leaks one fd per instance and multiplies log writes. Both handlers are
# created once per process and attached idempotently.

_file_handler: "logging.handlers.RotatingFileHandler | None" = None
_buf_handler: "logging.Handler | None" = None


def _get_file_handler() -> "logging.handlers.RotatingFileHandler | None":
    """Return the process-wide shell_infra.log handler, creating it once.

    Returns:
        The shared RotatingFileHandler, or None if it could not be created.

    Side effects:
        - creates ~/.config/sloughgpt/ and shell_infra.log on first call
    """
    global _file_handler
    if _file_handler is not None and not getattr(_file_handler, "closed", False):
        return _file_handler
    try:
        _log_dir = Path.home() / ".config" / "sloughgpt"
        _log_dir.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            str(_log_dir / "shell_infra.log"),
            maxBytes=10 * 1024 * 1024,
            backupCount=3,
        )
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s  %(message)s"
        ))
        handler.setLevel(logging.DEBUG)
        _file_handler = handler
    except Exception:
        return None
    return _file_handler


def _get_log_buffer_handler() -> "logging.Handler | None":
    """Return the process-wide log-buffer handler, creating it once.

    Returns:
        The shared LogBufferHandler, or None if the log buffer module is
        unavailable.

    Side effects:
        - attaches to the shared log buffer on first call
    """
    global _buf_handler
    if _buf_handler is not None and not getattr(_buf_handler, "closed", False):
        return _buf_handler
    try:
        from .log_buffer import get_log_buffer, LogBufferHandler
        _buf_handler = LogBufferHandler(get_log_buffer())
        _buf_handler.setLevel(logging.DEBUG)
    except Exception:
        return None
    return _buf_handler

# ── ANSI color constants (disabled via NO_COLOR env var) ─────────────

_COLOR_ENABLED = not os.environ.get("NO_COLOR")
if _COLOR_ENABLED:
    _C_CYAN = "\033[36m"
    _C_GREEN = "\033[32m"
    _C_YELLOW = "\033[33m"
    _C_RED = "\033[31m"
    _C_DIM = "\033[2m"
    _C_BOLD = "\033[1m"
    _C_RESET = "\033[0m"
else:
    _C_CYAN = _C_GREEN = _C_YELLOW = _C_RED = _C_DIM = _C_BOLD = _C_RESET = ""


def _color(text: str, code: str) -> str:
    """Wrap text in an ANSI color code, unless NO_COLOR is set."""
    return f"{code}{text}{_C_RESET}" if _COLOR_ENABLED and code else text

# ── readline (optional) ──────────────────────────────────────────────

_HAS_READLINE = False
try:
    import readline  # noqa: F401
    _HAS_READLINE = True
except ImportError:
    pass


# ── Completion cache fetchers (API-backed) ──────────────────────────

def _fetch_model_names() -> list[str]:
    """Fetch available model names from the API for tab completion."""
    import requests
    try:
        from .commands import get_api_base
        r = requests.get(f"{get_api_base()}/models", timeout=3)
        if r.status_code == 200:
            data = r.json()
            models = data if isinstance(data, list) else data.get("models", [])
            return sorted(set(m.get("name", m.get("id", "")) for m in models if isinstance(m, dict)))
    except Exception:
        pass
    return []

def _fetch_soul_names() -> list[str]:
    """Fetch soul names from the API for tab completion."""
    import requests
    try:
        from .commands import get_api_base
        r = requests.get(f"{get_api_base()}/souls", timeout=3)
        if r.status_code == 200:
            data = r.json()
            souls = data if isinstance(data, list) else data.get("souls", [])
            return sorted(set(s.get("name", "") for s in souls if isinstance(s, dict)))
    except Exception:
        pass
    return []

def _fetch_dataset_names() -> list[str]:
    """Fetch dataset names from the API for tab completion."""
    import requests
    try:
        from .commands import get_api_base
        r = requests.get(f"{get_api_base()}/datasets", timeout=3)
        if r.status_code == 200:
            data = r.json()
            datasets = data if isinstance(data, list) else data.get("datasets", [])
            return sorted(set(d.get("name", "") for d in datasets if isinstance(d, dict)))
    except Exception:
        pass
    return []

def _fetch_checkpoint_names() -> list[str]:
    """Fetch checkpoint names from the API for tab completion."""
    import requests
    try:
        from .commands import get_api_base
        r = requests.get(f"{get_api_base()}/auto-train/checkpoints", timeout=3)
        if r.status_code == 200:
            data = r.json()
            cps = data if isinstance(data, list) else data.get("checkpoints", [])
            return sorted(set(c.get("name", "") for c in cps if isinstance(c, dict)))
    except Exception:
        pass
    return []

_COMMAND_CACHE_FETCHERS: dict[str, callable] = {
    "load": _fetch_model_names,
    "unload": _fetch_model_names,
    "gen": _fetch_model_names,
    "protect": _fetch_model_names,
    "unprotect": _fetch_model_names,
    "switch": _fetch_soul_names,
    "datasets": _fetch_dataset_names,
    "dataset": _fetch_dataset_names,
    "checkpoints": _fetch_checkpoint_names,
}


# ── Output capture context manager ───────────────────────────────────


class _CaptureOutput:
    """Captures all write() output within a with-block. Test-only utility.

    If repl is provided, swaps repl.io to a MemoryIO buffer (proper capture).
    If no repl, falls back to capturing sys.stdout (legacy behavior).
    """

    def __init__(self, repl=None):
        self._repl = repl
        self._mem = None
        self._old_io = None
        self._buf = None
        self._old_stdout = None

    def __enter__(self):
        if self._repl is not None:
            from .io import MemoryIO
            self._mem = MemoryIO()
            self._old_io = self._repl.io
            self._old_console_io = getattr(self._repl.console, '_io', None)
            self._repl.io = self._mem
            if self._old_console_io is not None:
                self._repl.console._io = self._mem
        else:
            self._buf = io.StringIO()
            self._old_stdout = sys.stdout
            sys.stdout = self._buf
        return self

    def __exit__(self, *exc):
        if self._repl is not None and self._old_io is not None:
            self._repl.io = self._old_io
            if self._old_console_io is not None:
                self._repl.console._io = self._old_console_io
        elif self._old_stdout is not None:
            sys.stdout = self._old_stdout

    def getvalue(self) -> str:
        if self._mem is not None:
            return self._mem.get_output()
        if self._buf is not None:
            return self._buf.getvalue()
        return ""


# ── REPL ─────────────────────────────────────────────────────────────


class ShellREPL:
    """Interactive REPL with built-in commands and AI-assisted mode.

    For programmatic / TUI usage, call ``execute(line)`` directly —
    it returns ``(output, exit_code)`` without touching readline or
    the terminal.
    """

    def __init__(self, os: DaitRuntime, cmds: ShellCommands | None = None,
                 io: "ShellIO | None" = None, use_tui: bool | None = None):
        self.os = os
        self.cmds = cmds or ShellCommands()
        self.state = ShellState()
        self._history: list[str] = self.state.history[:]
        self._running = False
        # Line mode is the default interactive shell; the curses TUI is
        # opt-in via MAN_TUI=1 or the `tui` command / --tui flag.
        if use_tui is None:
            import os as _os
            try:
                use_tui = _os.environ.get("MAN_TUI") == "1"
            except (OSError, ValueError):
                use_tui = False
        self._use_tui = use_tui
        self._bg_threads: dict[int, threading.Thread] = {}
        self._next_bg_id = 1
        self._piped_input: str = ""
        self._aborted = False
        self._env: dict[str, str] = {
            "PS1": "\u03bb",
            "SHELL": "sloughgpt",
            "HOME": str(Path.home()),
            "TERM": "xterm-256color",
        }
        self._env.update(self.state.env)
        self._update_color_state()

        # I/O layer — swap for TUI via ``io=TuiIO()``
        from .io import ConsoleIO
        self.io = io or ConsoleIO()

        # Structured console output — tables, boxes, status, progress
        from .console import Console
        self.console = Console(self.io, has_readline=_HAS_READLINE)

        # Structured logger — inherit from domains.logging
        from domains.logging import ShellLogger, LogLevel
        self.log = ShellLogger("slo.shell.repl", level=LogLevel.DEBUG)

        # Log buffer — captures infra + API server logs for the console panel
        from .log_buffer import get_log_buffer, LogEntry
        self._log_buffer = get_log_buffer()
        _wrap_emit = self.log.emit
        def _buffered_emit(record):
            _wrap_emit(record)
            self._log_buffer.append(LogEntry(
                timestamp=record.timestamp,
                level=record.level.value.upper(),
                source=record.logger,
                message=record.message,
                context=dict(record.context),
            ))
        self.log.emit = _buffered_emit
        _log_buf_handler = _get_log_buffer_handler()
        if _log_buf_handler is not None:
            _slo_logger = logging.getLogger("slo")
            if _log_buf_handler not in _slo_logger.handlers:
                _slo_logger.addHandler(_log_buf_handler)
            # Also attach directly to child loggers that disable propagation during boot
            for _child_name in ("slo.kernel", "slo.shell.runtime", "slo.shell.init"):
                _child = logging.getLogger(_child_name)
                if _log_buf_handler not in _child.handlers:
                    _child.addHandler(_log_buf_handler)
        _log_dir = _audit_dir = Path.home() / ".config" / "sloughgpt"
        _log_dir.mkdir(parents=True, exist_ok=True)
        _file_handler = _get_file_handler()
        if _file_handler is not None:
            _slo_logger = logging.getLogger("slo")
            if _file_handler not in _slo_logger.handlers:
                _slo_logger.addHandler(_file_handler)

        self._log_buffer_bridge_setup = True

        # Audit logger — every command is logged to JSONL
        from .audit import get_shell_audit_logger
        self._audit = get_shell_audit_logger()

        # Permissions manager — gates destructive operations
        from .permissions import ShellPermissions
        self._perms = ShellPermissions()

        self._aliases: dict[str, str] = dict(self.state.aliases)
        self._aliases.update({
            "q": "exit", "quit": "exit", "h": "help",
            "?": "help",
            "jobs": "bg",
        })

        self._last_exit_code = 0
        self._cmd_count = 0
        self._dir_stack: list[str] = []
        self._chat_session_id: str | None = None
        self._chat_history: list[dict[str, str]] = []

        # Dynamic completion cache — TTL-based, shared with completion.py
        try:
            from core.completion import get_cache
            self._completion_cache_obj = get_cache()
        except ImportError:
            self._completion_cache_obj = None
        self._completion_cache: dict[str, tuple[float, list[str]]] = {}

        # External commands from commands/ directory
        from .cmds import discover as _discover
        self._ext_cmds = _discover()

        if _HAS_READLINE:
            self._setup_readline()

        self._load_rc()

    # ── Permission gate ─────────────────────────────────────────────

    def _check_permission(self, cmd: str, args_str: str, interactive: bool = True) -> bool:
        """Check if a command is allowed. Returns True if allowed, False if denied.

        In interactive mode, prompts the user when a command is blocked.
        In programmatic mode (execute()), silently denies.
        """
        from .permissions import Risk
        try:
            self._perms.check(cmd, args_str)
            return True
        except PermissionError as e:
            risk = self._perms.classify(cmd, args_str)
            if not interactive:
                self._print(f"  {_C_RED}Permission denied:{_C_RESET} {cmd} (risk={risk})")
                self._print(f"  Use `permit {cmd}` to grant, or `permit --all-{risk}` for all {risk} commands.")
                return False
            risk_colors = {
                Risk.SAFE: _C_GREEN,
                Risk.ELEVATED: _C_YELLOW,
                Risk.DANGEROUS: _C_RED,
                Risk.CRITICAL: _C_RED,
            }
            color = risk_colors.get(risk, _C_RED)
            self._print(f"  {_C_YELLOW}⚡ {cmd}{_C_RESET} requires {_C_BOLD}{color}{risk}{_C_RESET} permissions.")
            self._print(f"  Allow this command? {_C_DIM}[y/N/always]{_C_RESET}")
            try:
                answer = self.io.read("  > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = ""
            if answer == "always":
                self._perms.grant(cmd, persist=True)
                self._print(f"  {_C_GREEN}✓ Granted (persistent){_C_RESET} — {cmd} allowed for this and future sessions.")
                return True
            elif answer == "y":
                self._perms.grant(cmd)
                self._print(f"  {_C_GREEN}✓ Granted{_C_RESET} — {cmd} allowed this session.")
                return True
            else:
                self._print(f"  {_C_DIM}Denied{_C_RESET} — {cmd} skipped.")
                return False

    # ── Programmatic API (TUI / tests) ─────────────────────────────

    def execute(self, line: str) -> tuple[str, int]:
        """Execute a command line, return (output, exit_code).

        Does NOT touch readline or the terminal — safe for TUI usage.
        Still audit-logs every execution.
        """
        import time as _time
        from .io import MemoryIO

        if not line.strip():
            return "", 0

        mem = MemoryIO()
        old_io = self.io
        old_console_io = self.console._io
        self.io = mem
        self.console._io = mem
        old_print = self._print

        def _captured_print(*args, **kwargs):
            end = kwargs.get("end", "\n")
            text = " ".join(str(a) for a in args)
            self.io.write(text, end=end)

        self._print = _captured_print  # type: ignore

        t0 = _time.time()
        try:
            self._cmd_count += 1
            self._history.append(line)
            self.state.add_history(line)
            self.state.save()
            self._aborted = False
            self._piped_input = ""

            commands, is_bg, should_time = self._parse_pipeline(line)

            if not commands:
                self._print(f"  {_C_DIM}(empty pipeline){_C_RESET}")
                self._last_exit_code = 0
                self.io = old_io
                self.console._io = old_console_io
                self._print = old_print
                return mem.get_output(), 0

            if is_bg:
                if len(commands) > 1:
                    self._execute_background_tuples(commands)
                else:
                    self._execute_background(commands[0][0])
                self._audit.command(line, commands[0][0].split()[0] if commands[0][0] else "", line, 0, is_background=True, is_pipeline=len(commands) > 1)
                self.io = old_io
                self.console._io = old_console_io
                self._print = old_print
                return mem.get_output(), 0
            elif len(commands) > 1:
                self._execute_pipeline(commands, should_time=should_time)
                self._audit.command(line, "pipeline", line, self._last_exit_code, is_pipeline=True)
            else:
                raw_cmd, op = commands[0]
                expanded = self._expand_alias(raw_cmd)
                parts = expanded.split(maxsplit=1)
                cmd = parts[0].lower()
                args_str = parts[1] if len(parts) > 1 else ""
                handler = self.COMMANDS.get(cmd)
                ext_mod = self._ext_cmds.get(cmd) if handler is None else None
                if handler or ext_mod:
                    if not self._check_permission(cmd, args_str, interactive=False):
                        self._last_exit_code = 126
                        self._audit.command(line, cmd, args_str, 126, elapsed_ms=0, expanded=expanded)
                        self.io = old_io
                        self.console._io = old_console_io
                        self._print = old_print
                        return mem.get_output(), self._last_exit_code
                    try:
                        if ext_mod:
                            piped = self._piped_input if self._piped_input else ""
                            self._env["_piped_input"] = piped
                            self._env["_exec_fn"] = self._execute_single
                            c = Console(mem, has_readline=False)
                            self._last_exit_code = ext_mod.run([cmd] + (args_str.split() if args_str else []), c, self.cmds, self._env)
                            self._env.pop("_piped_input", None)
                        else:
                            handler(self, args_str)
                            self._last_exit_code = 0
                    except SystemExit as e:
                        self._last_exit_code = e.code if isinstance(e.code, int) else 1
                    except Exception as e:
                        self._print(f"  Error: {e}")
                        self._last_exit_code = 1
                        self._audit.error(line, repr(e))
                    elapsed_ms = (_time.time() - t0) * 1000
                    self._audit.command(line, cmd, args_str, self._last_exit_code, elapsed_ms=elapsed_ms, expanded=expanded)
                else:
                    suggestion = self._suggest_command(cmd)
                    msg = f"  Unknown command: {cmd}. Type `help`."
                    if suggestion:
                        msg += f" Did you mean `{suggestion}`?"
                    self._print(msg)
                    self._last_exit_code = 127
                    self._audit.unknown(cmd)
        except KeyboardInterrupt:
            self._aborted = True
            self._print("  Aborted")
        except Exception as e:
            self._print(f"  Error: {e}")
            self._last_exit_code = 1
            self._audit.error(line, repr(e))

        output = mem.get_output()
        self.io = old_io
        self.console._io = old_console_io
        self._print = old_print
        return output, self._last_exit_code

    def _rc_path(self) -> Path:
        return Path.home() / ".config" / "sloughgpt" / "rc"

    def _load_rc(self) -> None:
        """Execute ~/.config/sloughgpt/rc on startup (like bash .bashrc)."""
        rc = self._rc_path()
        if rc.is_file():
            for line_no, line in enumerate(rc.read_text().splitlines(), 1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                try:
                    cmds, is_bg, should_time = self._parse_pipeline(stripped)
                    if is_bg:
                        if len(cmds) > 1:
                            self._execute_background_tuples(cmds)
                        else:
                            self._execute_background(cmds[0][0])
                    elif len(cmds) > 1:
                        self._execute_pipeline(cmds, should_time=should_time)
                    else:
                        out = self._execute_single(cmds[0][0], "")
                except Exception as e:
                    logger.warning("rc line %d: %s", line_no, e, extra={"tag": "INFRA"})

    def _render_prompt(self) -> str:
        """Expand PS1 escapes: \\h=host, \\w=cwd, \\t=time, \\u=user, \\s=shell, \\#=cmd count,
        \\m=model, \\S=soul."""
        s = self._env.get("PS1", "\u03bb")
        s = s.replace("\\h", os.uname().nodename.split(".")[0])
        s = s.replace("\\w", os.getcwd().replace(str(Path.home()), "~"))
        s = s.replace("\\t", datetime.now().strftime("%H:%M:%S"))
        s = s.replace("\\u", os.environ.get("USER", "user"))
        s = s.replace("\\s", "sloughgpt")
        s = s.replace("\\#", str(self._cmd_count + 1))
        s = s.replace("\\n", "\n")
        s = s.replace("\\m", self._get_current_model())
        s = s.replace("\\S", self._get_current_soul())
        if self._last_exit_code != 0:
            s = f"{_C_RED}[{self._last_exit_code}]{_C_RESET} {s}"
        return s

    def _get_current_model(self) -> str:
        """Fetch loaded model name (cached 30s)."""
        now = time.monotonic()
        entry = self._completion_cache.get("__model__")
        if entry is not None:
            ts, val = entry
            if now - ts < 30.0:
                return val
        try:
            import requests
            from .config import get_api_base
            r = requests.get(f"{get_api_base()}/health", timeout=2)
            if r.status_code == 200:
                data = r.json()
                model = data.get("model", data.get("model_name", ""))
                if model:
                    val = str(model).split("/")[-1]
                    self._completion_cache["__model__"] = (now, val)
                    return val
        except Exception:
            pass
        return ""

    def _get_current_soul(self) -> str:
        """Fetch active soul name (cached 30s)."""
        now = time.monotonic()
        entry = self._completion_cache.get("__soul__")
        if entry is not None:
            ts, val = entry
            if now - ts < 30.0:
                return val
        try:
            import requests
            from .config import get_api_base
            r = requests.get(f"{get_api_base()}/souls/current", timeout=2)
            if r.status_code == 200:
                data = r.json()
                soul = data.get("name", "")
                if soul:
                    self._completion_cache["__soul__"] = (now, soul)
                    return soul
        except Exception:
            pass
        return ""

    def _update_color_state(self) -> None:
        """Sync ANSI color support with environment."""
        global _COLOR_ENABLED, _C_CYAN, _C_GREEN, _C_YELLOW, _C_RED, _C_DIM, _C_BOLD, _C_RESET
        no_color = self._env.get("NO_COLOR", "").strip().lower() in ("1", "true", "yes")
        if no_color == _COLOR_ENABLED:
            _COLOR_ENABLED = not no_color
            if _COLOR_ENABLED:
                _C_CYAN = "\033[36m"
                _C_GREEN = "\033[32m"
                _C_YELLOW = "\033[33m"
                _C_RED = "\033[31m"
                _C_DIM = "\033[2m"
                _C_BOLD = "\033[1m"
                _C_RESET = "\033[0m"
            else:
                _C_CYAN = _C_GREEN = _C_YELLOW = _C_RED = _C_DIM = _C_BOLD = _C_RESET = ""

    # ── readline setup ──────────────────────────────────────────────

    def _setup_readline(self) -> None:
        try:
            import readline
            histfile = Path.home() / ".config" / "sloughgpt" / ".shell_history"
            histfile.parent.mkdir(parents=True, exist_ok=True)

            # Truncate oversized history files (>10MB) to prevent slow startup.
            # Uses seek to find the last 5000 newlines without loading the
            # entire file into memory.
            _MAX_HIST_LINES = 5000
            _MAX_HIST_BYTES = 10 * 1024 * 1024
            if histfile.exists():
                try:
                    size = histfile.stat().st_size
                    if size > _MAX_HIST_BYTES:
                        with open(histfile, "rb") as f:
                            # Seek to find the position of the N-th line from end
                            f.seek(0, 2)
                            end = f.tell()
                            # Read last 2MB to find line boundaries
                            chunk_size = min(2 * 1024 * 1024, size)
                            f.seek(max(0, end - chunk_size))
                            tail = f.read()
                            # Count newlines and find the split point
                            nl_count = tail.count(b"\n")
                            if nl_count >= _MAX_HIST_LINES:
                                # Find the position of the (_MAX_HIST_LINES)th newline from end
                                pos = len(tail)
                                for _ in range(_MAX_HIST_LINES):
                                    pos = tail.rfind(b"\n", 0, pos)
                                    if pos < 0:
                                        break
                                # Rewrite file with only the tail
                                kept = tail[pos + 1:] if pos >= 0 else tail
                                with open(histfile, "wb") as wf:
                                    wf.write(kept)
                            # If fewer newlines than needed but file is huge,
                            # the file has very long lines — just keep last 2MB
                            elif nl_count < _MAX_HIST_LINES and chunk_size < size:
                                with open(histfile, "wb") as wf:
                                    wf.write(tail)
                            readline.set_history_length(_MAX_HIST_LINES)
                except Exception:
                    pass

            try:
                readline.read_history_file(str(histfile))
            except FileNotFoundError:
                pass
            readline.set_history_length(5000)
            import atexit
            atexit.register(lambda: readline.write_history_file(str(histfile)))
            readline.set_completer(self._complete)
            readline.parse_and_bind("tab: complete")
            readline.parse_and_bind('"\\C-r": reverse-search-history')
            readline.parse_and_bind('"\\C-s": forward-search-history')
        except Exception:
            pass

    def _complete(self, text: str, state: int) -> str | None:
        try:
            import readline
            line = readline.get_line_buffer()
        except Exception:
            line = text
        parts = line.strip().split()
        is_first_word = len(parts) <= 1 or line.endswith(" ")

        if is_first_word:
            candidates = list(self.COMMANDS.keys()) + list(self._ext_cmds.keys()) + list(self._aliases.keys())
        else:
            cmd = parts[0].lower()
            if cmd == "note" and len(parts) >= 2:
                sub = parts[1].lower() if len(parts) > 1 else ""
                if sub in ("show", "edit", "delete", "rm") and line.endswith(" "):
                    from notes import get_note_store
                    store = get_note_store(backend="mogdb")
                    candidates = [n.short_id for n in store.list_notes(limit=9999)]
                elif sub == "sprint" and line.endswith(" "):
                    from notes import get_note_store
                    store = get_note_store(backend="mogdb")
                    candidates = store.sprints()
                else:
                    candidates = self._complete_args_for(cmd)
            elif cmd == "finetuned" and len(parts) >= 2 and parts[1].lower() in ("load", "rm", "del", "delete") and line.endswith(" "):
                ft = self.cmds.finetuned_models()
                candidates = [m.get("model_name", "") for m in ft]
            else:
                candidates = self._complete_args_for(cmd)

        matches = [c for c in sorted(set(candidates)) if c.startswith(text)]
        try:
            return matches[state]
        except IndexError:
            return None

    def _complete_args_for(self, cmd: str) -> list[str]:
        """Return dynamic completion candidates for a given command.

        Uses CompletionCache (30s TTL) when available, falls back to local dict cache.
        """
        # Use CompletionCache if available (better error handling, stale data)
        if self._completion_cache_obj is not None:
            fetcher = _COMMAND_CACHE_FETCHERS.get(cmd)
            if fetcher:
                return self._completion_cache_obj.get(cmd, fetcher)
            # No API fetcher — fall through to path completion
            return self._complete_path("")

        # Fallback: local dict cache with 30s TTL
        now = time.monotonic()
        entry = self._completion_cache.get(cmd)
        if entry is not None:
            ts, values = entry
            if now - ts < 30.0:
                return values
        values = self._complete_args_for_uncached(cmd)
        self._completion_cache[cmd] = (now, values)
        return values

    def _complete_args_for_uncached(self, cmd: str) -> list[str]:
        """Fetch fresh completion candidates (no cache)."""
        try:
            if cmd in ("load", "unload", "gen", "protect", "unprotect"):
                models = self.cmds.models()
                return [m.get("name", m.get("id", "")) for m in models]
            if cmd in ("switch",):
                souls = self.cmds.souls()
                return [s.get("name", "") for s in souls]
            if cmd in ("datasets",):
                ds = self.cmds.datasets()
                return [d.get("name", "") for d in ds]
            if cmd in ("checkpoints",):
                cps = self.cmds.checkpoints()
                return [cp.get("name", "") for cp in cps]
            if cmd in ("finetuned",):
                return ["load", "rm", "del", "delete"]
            if cmd == "train":
                return ["status", "follow", "stop", "distill", "hf", "auto", "load", "del"]
            if cmd in ("permit", "deny"):
                from .permissions import _DANGEROUS, _CRITICAL
                candidates = sorted(_DANGEROUS | _CRITICAL) + ["--persist"]
                if cmd == "permit":
                    candidates.append("--all-dangerous")
                return candidates
            if cmd == "note":
                return ["new", "list", "show", "edit", "delete", "search", "today", "export", "tags", "status", "sprint", "timeline"]
        except Exception:
            pass
        return self._complete_path("")

    def _complete_path(self, prefix: str) -> list[str]:
        """Return matching file/directory paths for tab completion."""
        # VFS-aware completion for /dev/ and /proc/
        if prefix.startswith("/dev") or prefix.startswith("/proc"):
            try:
                vfs = self.os.vfs
                if vfs:
                    parent = prefix.rsplit("/", 1)[0] or prefix
                    partial = prefix.rsplit("/", 1)[1] if "/" in prefix else ""
                    if parent == prefix:
                        parent = prefix.rstrip("/")
                        partial = ""
                    entries = vfs.listdir(parent) if parent else vfs.listdir("/dev")
                    if entries is not None:
                        matches = [e + "/" if vfs.isdir(parent + "/" + e if parent else "/dev/" + e) else e for e in entries if not e.startswith(".") and e.startswith(partial)]
                        return sorted(matches)
            except Exception:
                pass
        if not prefix or prefix == "." or prefix == "..":
            search_dir = Path(".")
            partial = prefix
        else:
            p = Path(prefix)
            if prefix.endswith("/"):
                search_dir = p
                partial = ""
            else:
                search_dir = p.parent
                partial = p.name
            if not search_dir.exists():
                return []
        try:
            candidates = []
            for entry in search_dir.iterdir():
                name = entry.name
                if name.startswith("."):
                    continue
                if name.startswith(partial):
                    suffix = "/" if entry.is_dir() else ""
                    candidates.append(str(entry) + suffix)
            return sorted(candidates)
        except PermissionError:
            return []

    # ── I/O helpers ─────────────────────────────────────────────────

    def _print(self, *args, **kwargs) -> None:
        end = kwargs.get("end", "\n")
        text = " ".join(str(a) for a in args)
        self.console.write(text, end=end)

    def _table(self, rows: list[list[str]], header: list[str] | None = None,
               separator_after_header: bool = True) -> None:
        self.console.table(rows, header, separator_after_header)

    def _box(self, text: str, width: int | None = None) -> None:
        self.console.box(text, width)

    def _status(self, kind: str, message: str, detail: str = "") -> None:
        self.console.status(kind, message, detail)

    def _kvlist(self, items: list[tuple[str, str]]) -> None:
        self.console.kvlist(items)

    def _log_ok(self, msg: str, **ctx) -> None:
        """Log a success message (green checkmark)."""
        from domains.logging import LogLevel
        self.log.emit(self.log._make_record(LogLevel.INFO, msg, ctx))

    def _log_warn(self, msg: str, **ctx) -> None:
        """Log a warning (yellow exclamation)."""
        self.log.warning(msg, **ctx)

    def _log_error(self, msg: str, **ctx) -> None:
        """Log an error (red cross)."""
        self.log.error(msg, **ctx)

    def _log_step(self, msg: str, **ctx) -> None:
        """Log a step/action (cyan arrow)."""
        self.log.info(msg, **ctx)

    def _print_header(self) -> None:
        self._print(f"  Type {_C_YELLOW}`help`{_C_RESET} for commands, {_C_YELLOW}`exit`{_C_RESET} to quit, {_C_YELLOW}`ai <query>`{_C_RESET} for AI mode")

    def _format_table(self, rows: list[list[str]], header: list[str] | None = None) -> str:
        if not rows:
            return "(empty)"
        cols = max(len(r) for r in rows)
        if header:
            cols = max(cols, len(header))
        widths = [0] * cols
        for row in rows:
            for i, cell in enumerate(row):
                if i < cols:
                    widths[i] = max(widths[i], len(str(cell)))
        if header:
            for i, cell in enumerate(header):
                if i < cols:
                    widths[i] = max(widths[i], len(str(cell)))
        lines = []
        fmt = "  ".join("{{:<{}}}".format(w) for w in widths)
        if header:
            lines.append(fmt.format(*header))
            lines.append("  ".join("─" * w for w in widths))
        for row in rows:
            padded = list(row) + [""] * (cols - len(row))
            lines.append(fmt.format(*padded))
        return "\n".join(lines)

    def _dump_json(self, obj: Any) -> str:
        return json.dumps(obj, indent=2, default=str)

    def _spinner_call(self, label: str, fn, ok_msg: str | None = ""):
        """
        Standard loading-state wrapper for any operation that makes API calls
        or has a timeout. Shows a spinner animation while *fn* runs, then
        replaces it with a success or cleared line.

        Args:
            label:  Spinner label text (e.g. "Fetching models", "Generating").
            fn:     Zero-arg callable wrapping the blocking operation.
            ok_msg: Success message shown after completion.
                    - ``None``  → spinner line is cleared silently (for list/table commands).
                    - ``""``    → uses *label* as the success message.
                    - ``str``   → custom success message.

        Every command that calls the HTTP API MUST use _spinner_call or an
        equivalent loading indicator — this is a codebase standard.
        """
        with self.console.spinner(label) as s:
            result = fn()
        if ok_msg is not None:
            s.ok(ok_msg or label)
        return result

    def _expand_vars(self, text: str) -> str:
        """Replace $VAR, ${VAR}, and $? with values."""
        text = text.replace("$?", str(self._last_exit_code))
        def _repl(m: re.Match) -> str:
            name = m.group(1) or m.group(2) or ""
            return self._env.get(name, m.group(0))
        result = re.sub(r"\$\{(\w+)\}|\$(\w+)", _repl, text)
        return result

    def _expand_cmd_subst(self, text: str) -> str:
        """Replace $(command) with the captured output of that command."""
        pattern = r"\$\(([^()]+)\)"
        while re.search(pattern, text):
            def _repl(m: re.Match) -> str:
                inner = m.group(1).strip()
                out = self._execute_single(inner, "")
                return out.rstrip("\n")
            text = re.sub(pattern, _repl, text, count=1)
        return text

    def _expand_history(self, text: str) -> str:
        """Expand ! history references: !! !$ !n !-n !-n$ etc."""
        result = text

        # !! → last command
        result = re.sub(r'(?<!\w)!!(?!\w)', lambda m: self._history[-1] if self._history else "!!", result)

        # !$ → last arg of last command
        def _last_arg(m: re.Match) -> str:
            if not self._history:
                return m.group(0)
            parts = self._history[-1].split()
            return parts[-1] if len(parts) > 1 else parts[0]
        result = re.sub(r'(?<!\w)!\$(?!\w)', _last_arg, result)

        # !:N → Nth arg of last command (0-indexed)
        def _nth_arg(m: re.Match) -> str:
            n = int(m.group(1))
            if not self._history:
                return m.group(0)
            parts = self._history[-1].split()
            return parts[n] if n < len(parts) else m.group(0)
        result = re.sub(r'!:(0|[1-9]\d*)', _nth_arg, result)

        # !* → all args of last command
        def _all_args(m: re.Match) -> str:
            if not self._history:
                return m.group(0)
            parts = self._history[-1].split()
            return " ".join(parts[1:]) if len(parts) > 1 else ""
        result = re.sub(r'(?<!\w)!\*(?!\w)', _all_args, result)

        # !-n → command n from end
        def _neg_history(m: re.Match) -> str:
            n = int(m.group(1))
            if not self._history or n > len(self._history):
                return m.group(0)
            return self._history[-n]
        result = re.sub(r'!-(\d+)(?!\w)', _neg_history, result)

        # !n → command #n (must be after !-n so it doesn't match first)
        def _pos_history(m: re.Match) -> str:
            n = int(m.group(1))
            if n < 1 or n > len(self._history):
                return m.group(0)
            return self._history[n - 1]
        result = re.sub(r'(?<!\w)!(\d+)(?!\w)', _pos_history, result)

        return result

    def _expand_globs(self, text: str) -> str:
        """Expand glob patterns (*, ?, []) in command arguments.
        Only expands patterns with actual file matches."""
        if not glob.has_magic(text):
            return text

        # Tokenize respecting quotes
        tokens = []
        i = 0
        while i < len(text):
            if text[i] in ('"', "'"):
                quote = text[i]
                j = i + 1
                while j < len(text) and text[j] != quote:
                    if text[j] == '\\':
                        j += 1
                    j += 1
                tokens.append(text[i:j+1])
                i = j + 1
            elif text[i] == ' ':
                tokens.append(' ')
                i += 1
            else:
                j = i
                while j < len(text) and text[j] not in (' ', '"', "'"):
                    j += 1
                tokens.append(text[i:j])
                i = j

        # Expand glob patterns (but not quoted tokens or commands)
        expanded = []
        for t in tokens:
            if t == ' ':
                expanded.append(t)
            elif t.startswith(("'", '"')):
                expanded.append(t)
            elif glob.has_magic(t):
                matches = sorted(glob.glob(t))
                if matches:
                    # Quote filenames with spaces
                    quoted = []
                    for m in matches:
                        if ' ' in m:
                            quoted.append(f"'{m}'")
                        else:
                            quoted.append(m)
                    expanded.append(" ".join(quoted))
                else:
                    expanded.append(t)
            else:
                expanded.append(t)

        return "".join(expanded)

    def _expand_alias(self, line: str) -> str:
        parts = line.strip().split(maxsplit=1)
        if not parts:
            return line
        cmd = parts[0].lower()
        if cmd in self._aliases:
            rest = parts[1] if len(parts) > 1 else ""
            return self._expand_vars(f"{self._aliases[cmd]} {rest}").strip()
        return self._expand_vars(line.strip())

    # ── Pipeline + background execution ─────────────────────────────

    def _parse_pipeline(self, line: str):
        """Return (commands, is_background, should_time) where commands is a list of (cmd_str, operator) tuples.

        Operators: None (pipe to next), '&&', '||', ';' (chain), '|' (pipe).
        """
        bg = False
        should_time = False
        stripped = line.rstrip()
        if stripped.endswith("&"):
            bg = True
            stripped = stripped[:-1].rstrip()
        if stripped.startswith("time "):
            should_time = True
            stripped = stripped[5:].lstrip()

        # Split by chaining operators (&&, ||, ;) and pipes
        chain_re = re.compile(r'(&&|\|\||;|\|)')
        tokens = []
        pos = 0
        depth = 0
        in_quote = None
        for i, ch in enumerate(stripped):
            if ch in ('"', "'") and (i == 0 or stripped[i-1] != '\\'):
                if in_quote is None:
                    in_quote = ch
                elif in_quote == ch:
                    in_quote = None
            if in_quote:
                continue
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            if depth == 0:
                m = chain_re.match(stripped, i)
                if m:
                    tokens.append((stripped[pos:i].strip(), m.group(1)))
                    pos = m.end()
        if pos < len(stripped):
            tokens.append((stripped[pos:].strip(), None))

        # Group by pipes within each chain segment
        commands = []
        for seg, op in tokens:
            pipe_parts = self._split_pipe(seg)
            for pp in pipe_parts[:-1]:
                commands.append((pp, '|'))
            commands.append((pipe_parts[-1], op))

        return commands, bg, should_time

    @staticmethod
    def _split_pipe(seg: str) -> list[str]:
        parts = []
        cur: list[str] = []
        in_quote: str | None = None
        for ch in seg:
            if ch in ("'", '"'):
                if in_quote is None:
                    in_quote = ch
                elif in_quote == ch:
                    in_quote = None
            if ch == "|" and in_quote is None:
                parts.append("".join(cur).strip())
                cur = []
            else:
                cur.append(ch)
        parts.append("".join(cur).strip())
        return parts

    def _strip_redirection(self, raw_args: str):
        """Strip '> file' or '>> file' from the end of args. Returns (cleaned_args, redirect_path, append_mode)."""
        stripped = raw_args.rstrip()
        redirect_path = None
        append_mode = False
        m = re.search(r"(>>?)\s+(\S+)\s*$", stripped)
        if m:
            redirect_path = m.group(2)
            append_mode = m.group(1) == ">>"
            stripped = stripped[:m.start()].rstrip()
        return stripped, redirect_path, append_mode

    def _parse_inline_env(self, raw: str):
        """Parse leading NAME=VALUE assignments before a command.
        Returns (env_updates, remaining_args)."""
        env_updates = {}
        rest = raw.strip()
        while rest:
            m = re.match(r"(\w+)=(\S+)\s*", rest)
            if m and not self.COMMANDS.get(m.group(1)):
                env_updates[m.group(1)] = m.group(2).strip("\"'")
                rest = rest[m.end():]
            else:
                break
        return env_updates, rest.strip()

    def _execute_single(self, raw: str, piped_input: str = "") -> str:
        """Execute a single command (or pipeline segment) and return its output."""
        expanded = self._expand_alias(raw)
        expanded = self._expand_cmd_subst(expanded)
        expanded = self._expand_history(expanded)
        expanded = self._expand_globs(expanded)
        inline_env, cleaned = self._parse_inline_env(expanded)
        cleaned, redirect_path, append_mode = self._strip_redirection(cleaned)

        # Apply inline env vars BEFORE variable expansion so $VAR picks them up
        old_env = {}
        for k, v in inline_env.items():
            old_env[k] = self._env.get(k)
            self._env[k] = v

        # Re-expand $VAR now that inline env is active
        cleaned = self._expand_vars(cleaned)
        parts = cleaned.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        self._piped_input = piped_input
        handler = self.COMMANDS.get(cmd)
        ext_mod = self._ext_cmds.get(cmd) if handler is None else None

        if handler is None and ext_mod is None:
            # Fallback: try as a system binary
            import shutil, subprocess, shlex
            binary = shutil.which(cmd)
            if binary:
                from .io import MemoryIO as _MemIO
                cap = _MemIO()
                old_io = self.io
                old_console_io = self.console._io
                self.io = cap
                self.console._io = cap
                try:
                    shell_args = shlex.split(args) if args else []
                    stdin_data = piped_input if isinstance(piped_input, bytes) else (piped_input.encode() if piped_input else None)
                    sub_env = {**os.environ, **self._env}
                    result = subprocess.run(
                        [binary] + shell_args,
                        input=stdin_data,
                        capture_output=True,
                        timeout=120,
                        env=sub_env,
                    )
                    out_text = result.stdout.decode(errors="replace") if result.stdout else ""
                    err_text = result.stderr.decode(errors="replace") if result.stderr else ""
                    if out_text:
                        self._print(out_text.rstrip("\n"))
                    if err_text:
                        self._print(err_text.rstrip("\n"))
                    self._last_exit_code = result.returncode
                except subprocess.TimeoutExpired:
                    self._print(f"  Command timed out: {cmd}")
                    self._last_exit_code = 124
                except Exception as e:
                    self._print(f"  Error: {e}")
                    self._last_exit_code = 1
                finally:
                    self.io = old_io
                    self.console._io = old_console_io

                # Handle redirect
                cap_out = cap.get_output()
                if redirect_path:
                    vfs = self.os.vfs
                    if vfs and (redirect_path.startswith("/dev/") or redirect_path.startswith("/proc/")):
                        result = vfs.write(redirect_path, cap_out)
                        if result:
                            self._print(result)
                    else:
                        mode = "a" if append_mode else "w"
                        try:
                            with open(os.path.expanduser(redirect_path), mode) as f:
                                f.write(cap_out)
                        except OSError:
                            self._last_exit_code = 1
                    cap_out = ""

                for k in inline_env:
                    if old_env[k] is None:
                        self._env.pop(k, None)
                    else:
                        self._env[k] = old_env[k]
                self._piped_input = ""
                return cap_out

            self._piped_input = ""
            for k in inline_env:
                if old_env[k] is None:
                    self._env.pop(k, None)
                else:
                    self._env[k] = old_env[k]
            suggestion = self._suggest_command(cmd)
            msg = f"  Unknown command: {cmd}. Type `help`."
            if suggestion:
                msg += f" Did you mean `{suggestion}`?"
            self._last_exit_code = 127
            return msg + "\n"

        if not self._check_permission(cmd, args, interactive=False):
            self._last_exit_code = 126
            for k in inline_env:
                if old_env[k] is None:
                    self._env.pop(k, None)
                else:
                    self._env[k] = old_env[k]
            self._piped_input = ""
            return f"  Permission denied: {cmd} (use `permit {cmd}` to grant)\n"

        from .io import MemoryIO as _MemIO
        cap = _MemIO()
        old_io = self.io
        old_console_io = self.console._io
        self.io = cap
        self.console._io = cap
        try:
            try:
                if ext_mod:
                    argv = [cmd] + (args.split() if args else [])
                    c = Console(self.io, has_readline=_HAS_READLINE)
                    if piped_input:
                        self._env["_piped_input"] = piped_input
                    self._env["_exec_fn"] = self._execute_single
                    self._last_exit_code = ext_mod.run(argv, c, self.cmds, self._env)
                    self._env.pop("_piped_input", None)
                else:
                    self._last_exit_code = 0
                    handler(self, args)
            except SystemExit as e:
                self._last_exit_code = e.code if isinstance(e.code, int) else 1
            except Exception as e:
                self._print(f"  Error: {e}")
                self._last_exit_code = 1
        finally:
            self.io = old_io
            self.console._io = old_console_io

        for k in inline_env:
            if old_env[k] is None:
                self._env.pop(k, None)
            else:
                self._env[k] = old_env[k]

        self._piped_input = ""
        output = cap.get_output()

        if redirect_path:
            vfs = self.os.vfs
            if vfs and (redirect_path.startswith("/dev/") or redirect_path.startswith("/proc/")):
                result = vfs.write(redirect_path, output)
                if result:
                    self._print(result)
                return ""
            mode = "a" if append_mode else "w"
            try:
                with open(os.path.expanduser(redirect_path), mode) as f:
                    f.write(output)
                return ""
            except OSError as e:
                self._last_exit_code = 1
                return f"  Error writing to {redirect_path}: {e}\n"

        return output

    def _suggest_command(self, bad_cmd: str) -> str | None:
        """Suggest a close command match via difflib (excludes short aliases)."""
        import difflib
        all_cmds = list(self.COMMANDS.keys()) + list(self._ext_cmds.keys())
        matches = difflib.get_close_matches(bad_cmd, all_cmds, n=1, cutoff=0.6)
        return matches[0] if matches else None

    def _execute_pipeline(self, commands: list, should_time: bool = False) -> None:
        """Execute a pipeline of chained commands with &&, ||, ;, and | operators.

        Args:
            commands: list of (cmd_str, operator) tuples
        """
        if not commands:
            return
        piped = ""
        t0 = None
        if should_time:
            import time as _time
            t0 = _time.time()

        for i, (raw, op) in enumerate(commands):
            is_last = i == len(commands) - 1

            # Check if we should skip based on previous exit code
            if op == '&&' and self._last_exit_code != 0:
                continue
            if op == '||' and self._last_exit_code == 0:
                continue

            out = self._execute_single(raw, piped)
            if is_last or op != '|':
                self._print(out, end="")
            piped = out if op == '|' else ""

        if should_time and t0 is not None:
            import time as _time
            elapsed = _time.time() - t0
            self._print(f"{_C_DIM}  [{elapsed:.2f}s]{_C_RESET}")

    def _execute_background(self, raw: str) -> None:
        """Execute a command in a background thread."""
        bg_id = self._next_bg_id
        self._next_bg_id += 1

        def _run():
            try:
                out = self._execute_single(raw, "")
                with threading.Lock():
                    self._print(f"\n[bg-{bg_id}] {out}", end="")
            except Exception as e:
                self._print(f"\n[bg-{bg_id}] Error: {e}")

        t = threading.Thread(target=_run, daemon=True, name=f"shell-bg-{bg_id}")
        t.start()
        self._bg_threads[bg_id] = t
        self._print(f"  [bg-{bg_id}] {raw}")

    def _execute_background_tuples(self, commands: list) -> None:
        """Execute chained commands in a background thread."""
        bg_id = self._next_bg_id
        self._next_bg_id += 1

        def _run():
            try:
                self._execute_pipeline(commands)
            except Exception as e:
                self._print(f"\n[bg-{bg_id}] Error: {e}")

        t = threading.Thread(target=_run, daemon=True, name=f"shell-bg-{bg_id}")
        t.start()
        self._bg_threads[bg_id] = t
        cmds_str = " | ".join(c[0] for c in commands)
        self._print(f"  [bg-{bg_id}] {cmds_str}")

    # ── Pipe-filter commands ────────────────────────────────────────

    # ── Alias commands ──────────────────────────────────────────────

    def _cmd_alias(self, args: str = "") -> None:
        if not args:
            for name, cmd in sorted(self._aliases.items()):
                self._print(f"  {name}={cmd}")
            return
        if "=" in args:
            name, _, command = args.partition("=")
            name = name.strip()
            command = command.strip()
            self._aliases[name] = command
            self.state.set_alias(name, command)
            self.state.save()
        else:
            cmd = self._aliases.get(args.strip())
            if cmd:
                self._print(f"  {args.strip()}={cmd}")
            else:
                self._print(f"  No alias for '{args.strip()}'")

    def _cmd_unalias(self, args: str = "") -> None:
        if not args:
            self._print("  Usage: unalias <name>")
            return
        name = args.strip()
        if name in self._aliases:
            del self._aliases[name]
            self.state.unset_alias(name)
            self.state.save()
            self._print(f"  Removed alias '{name}'")
        else:
            self._print(f"  No alias '{name}'")

    def _cmd_export_state(self, args: str = "") -> None:
        self._print(self._dump_json(self.state.to_dict()))

    def _cmd_set(self, args: str = "") -> None:
        """Set or show environment variables."""
        if not args:
            for k, v in sorted(self._env.items()):
                self._print(f"  {k}={v}")
            return
        if "=" in args:
            name, _, value = args.partition("=")
            name = name.strip()
            value = value.strip().strip("\"'")
            self._env[name] = value
            if name == "NO_COLOR":
                self._update_color_state()
            self.state.set_env(name, value)
            self.state.save()
        else:
            value = self._env.get(args.strip())
            if value is not None:
                self._print(f"  {args.strip()}={value}")
            else:
                self._print(f"  {args.strip()} not set")

    def _cmd_source(self, args: str = "") -> None:
        """Execute commands from a file (like bash source/.)."""
        if not args:
            self._print("  Usage: source <file>   or  . <file>")
            return
        path = os.path.expanduser(args.strip())
        try:
            with open(path) as f:
                for line_no, line in enumerate(f, 1):
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#"):
                        continue
                    try:
                        cmds, is_bg, should_time = self._parse_pipeline(stripped)
                        if is_bg:
                            if len(cmds) > 1:
                                self._execute_background_tuples(cmds)
                            else:
                                self._execute_background(cmds[0][0])
                        elif len(cmds) > 1:
                            self._execute_pipeline(cmds, should_time=should_time)
                        else:
                            expanded = self._expand_alias(cmds[0][0])
                            out = self._execute_single(expanded, "")
                            if out:
                                self._print(out, end="")
                    except Exception as e:
                        self._print(f"  Error at line {line_no}: {e}")
        except OSError as e:
            self._print(f"  Error reading {path}: {e}")

    def _cmd_py(self, args: str = "") -> None:
        """Evaluate a Python expression and print the result.

        Sandboxed: only safe builtins and whitelisted modules available.
        Every evaluation is audit-logged.
        """
        if not args:
            self._print("  Usage: py <expression>")
            self._print("  Example: py 2 + 2")
            self._print("  Example: py [i*i for i in range(5)]")
            self._print("  Example: py __import__('json').dumps({'a': 1})")
            return

        # Restricted __import__ — only safe stdlib modules
        _SAFE_MODULES = frozenset({
            "math", "json", "datetime", "time", "re", "collections",
            "itertools", "functools", "operator", "string", "textwrap",
            "statistics", "decimal", "fractions", "random", "uuid",
            "hashlib", "base64", "binascii", "struct", "codecs",
            "unicodedata", "enum", "dataclasses", "typing", "copy",
            "pprint", "array", "heapq", "bisect", "graphlib",
        })

        def _safe_import(name: str, *args: Any, **kwargs: Any) -> Any:
            root = name.split(".")[0]
            if root not in _SAFE_MODULES:
                raise ImportError(
                    f"module {name!r} is not allowed in py. "
                    f"Allowed: {', '.join(sorted(_SAFE_MODULES))}"
                )
            return __import__(name, *args, **kwargs)

        safe_builtins = {
            "abs": abs, "all": all, "any": any, "bool": bool,
            "chr": chr, "dict": dict, "dir": dir, "enumerate": enumerate,
            "filter": filter, "float": float, "format": format,
            "frozenset": frozenset, "getattr": getattr,
            "hasattr": hasattr, "hash": hash, "hex": hex,
            "int": int, "isinstance": isinstance, "issubclass": issubclass,
            "iter": iter, "len": len, "list": list, "map": map,
            "max": max, "min": min, "next": next, "oct": oct, "ord": ord,
            "pow": pow, "print": print, "property": property,
            "range": range, "repr": repr, "reversed": reversed,
            "round": round, "set": set, "slice": slice, "sorted": sorted,
            "str": str, "sum": sum, "super": super, "tuple": tuple,
            "type": type, "zip": zip, "__import__": _safe_import,
        }

        exit_code = 0
        result_repr = ""
        try:
            result = eval(args, {"__builtins__": safe_builtins})
            result_repr = repr(result)
            self._print(result_repr)
        except Exception as e:
            exit_code = 1
            result_repr = f"Error: {e}"
            self._print(f"  {result_repr}")

        # Audit-log every evaluation
        self._audit.eval(args, result_repr, exit_code)

    def _cmd_bg(self, args: str = "") -> None:
        if not self._bg_threads:
            self._print("  No background processes")
            return
        for bg_id, t in sorted(self._bg_threads.items()):
            alive = t.is_alive()
            self._print(f"  [bg-{bg_id}] {'running' if alive else 'done'}")

    def _cmd_fg(self, args: str = "") -> None:
        """Bring a background process to the foreground."""
        if not args:
            self._print("  Usage: fg <id>")
            self._print("  Use `bg` or `jobs` to list running IDs")
            return
        bg_id_s = args.strip()
        if not bg_id_s.isdigit():
            self._print(f"  Invalid id: {args}")
            return
        bg_id = int(bg_id_s)
        t = self._bg_threads.get(bg_id)
        if t is None:
            self._print(f"  No background process [bg-{bg_id}]")
            return
        if not t.is_alive():
            self._print(f"  [bg-{bg_id}] already done")
            return
        self._print(f"  Waiting for [bg-{bg_id}]...")
        t.join(timeout=600)
        if t.is_alive():
            self._print(f"  [bg-{bg_id}] still running (use `bg` to check)")

    def _cmd_watch(self, args: str = "") -> None:
        """Run a command repeatedly every N seconds."""
        parts = args.strip().split(maxsplit=1)
        if len(parts) < 2:
            self._print("  Usage: watch <interval_sec> <command>")
            self._print("  Example: watch 2 health")
            return
        try:
            interval = float(parts[0])
        except ValueError:
            self._print(f"  Invalid interval: {parts[0]}")
            return
        cmd = parts[1]
        import time as _time
        iteration = 1
        self._print(f"  Watching every {interval}s — Ctrl+C to stop")
        try:
            while True:
                out = self._execute_single(cmd, "")
                self._print(f"{_C_DIM}--- iteration {iteration} ---{_C_RESET}")
                self._print(out, end="")
                iteration += 1
                _time.sleep(interval)
        except KeyboardInterrupt:
            self._print(f"  {_C_DIM}Stopped{_C_RESET}")

    def _cmd_export(self, args: str = "") -> None:
        """POSIX-style export: export NAME=VALUE or export NAME."""
        if not args:
            self._cmd_set("")
            return
        if "=" in args:
            self._cmd_set(args)
        else:
            name = args.strip()
            value = self._env.get(name)
            if value is not None:
                self._print(f"  {name}={value}")
            else:
                self._print(f"  {name} not set")

    # ── Existing command handlers ───────────────────────────────────

    @staticmethod
    def _group_ext_cmds(ext_cmds: dict[str, CmdModule]) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for name, m in ext_cmds.items():
            h = getattr(m, "help", "") or ""
            groups.setdefault(h, []).append(name)
        for names in groups.values():
            names.sort()
        return dict(sorted(groups.items(), key=lambda x: x[1][0]))

    def _cmd_help(self, args: str = "") -> None:
        if args:
            if args == "brief":
                self._print("""
  Most-used commands (help <cmd> for details, help for full list):

  models / load / unload  Model management
  souls / switch        Soul personality management
  gen / chat / ai       Inference and natural language
  health / status       System health and info
  boot / shutdown       Shell lifecycle
  api                   API server lifecycle (start/stop/status)
  svc                   Service management
  devices / lsdev       AI device nodes (/dev/llm, /dev/embedding, /dev/knowledge)
  asm                   Virtual machine
  remember / recall     Knowledge base
  datasets / tokenizer  Data and tokenizer utilities
  procs / kill / bg / fg  Process management
  py                    Shell utilities
  cd / pwd / echo / ls / cat / mkdir / rm / touch / cp / mv / head / tail / wc   Filesystem operations
  history / fc          Command history
  help <cmd>            Help for a specific command
  exit / q / quit       Exit shell

  Pipe features: |  &  >  >>  $(...)  $?  $VAR
""")
                return
            cmd_help = {
                "help": "  help [cmd]  — Show this help or help for a specific command",
                "exit": "  exit | q | quit  — Exit the shell",
                "cd": '  cd [dir]  — Change directory (default: ~, - for previous)',
                "pwd": "  pwd  — Print working directory",
                "echo": "  echo [text...]  — Print text to stdout",
                "ls": "  ls [dir]  — List directory contents (VFS-aware: shows /dev/*, /proc/*)",
                "cat": "  cat <file>  — Print file contents (VFS-aware: reads /dev/*, /proc/*)",
                "mkdir": "  mkdir <dir>  — Create a directory",
                "rm": "  rm [-rf] <path>  — Remove files/directories (permission-gated)",
                "touch": "  touch <file> [file...]  — Create empty files or update timestamps",
                "cp": "  cp <src> <dst>  — Copy files/directories",
                "mv": "  mv <src> <dst>  — Move or rename files",
                "head": "  head [-N] <file>  — Output first N lines (default 10, VFS-aware)",
                "tail": "  tail [-N] <file>  — Output last N lines (default 10, VFS-aware)",
                "wc": "  wc <file>  — Count lines, words, characters (VFS-aware)",
                "grep": "  grep [-i] [-v] <pattern> [file]  — Search for patterns (VFS-aware, supports pipes)",
                "sort": "  sort [-r] [-n] [-u] [file]  — Sort lines (supports pipes)",
                "uniq": "  uniq [file]  — Remove adjacent duplicate lines (supports pipes)",
                "find": "  find [dir] -name <pattern>  — Search for files by name",
                "tee": "  tee [-a] <file>  — Copy stdin to file and stdout",
                "xargs": "  xargs [-n N] <cmd>  — Build and execute command from stdin",
                "chmod": "  chmod <mode> <file>  — Change file permissions (octal)",
                "du": "  du [-h] [path...]  — Estimate disk usage",
                "diff": "  diff <file1> <file2>  — Compare files line by line",
                "stat": "  stat <path>  — Display file metadata",
                "cut": "  cut -f<N> [-d<delim>] [file]  — Cut fields from lines (supports pipes)",
                "tr": "  tr <set1> <set2>  — Translate characters (piped input)",
                "seq": "  seq [first [inc]] last  — Generate number sequence",
                "nl": "  nl [file]  — Number lines of a file or piped input",
                "fold": "  fold [-w width] [file]  — Wrap long lines at a specified width",
                "tac": "  tac [file]  — Reverse lines (cat backwards)",
                "env": "  env  — Print environment variables",
                "printenv": "  printenv  — Print environment variables",
                "yes": "  yes [string]  — Repeatedly output a line",
                "realpath": "  realpath <path>  — Resolve path to absolute",
                "dirname": "  dirname <path>  — Strip last component from file path",
                "basename": "  basename <path> [suffix]  — Strip directory from file path",
                "nproc": "  nproc  — Print number of CPUs",
                "hostname": "  hostname  — Print system hostname",
                "uname": "  uname [-a] [-srm]  — Print system information",
                "shuf": "  shuf [file]  — Shuffle lines randomly",
                "rev": "  rev [file]  — Reverse characters in each line",
                "paste": "  paste <file1> [file2 ...]  — Merge lines of files side by side",
                "comm": "  comm <file1> <file2>  — Compare two sorted files line by line",
                "test": "  test <expr>  — Evaluate conditional expression (sets $? 0=true 1=false)",
                "[": "  [ <expr> ]  — Synonym for test",
                "printf": "  printf <format> [args...]  — Format and print data (%s %d %f \\n \\t)",
                "expand": "  expand [file]  — Convert tabs to spaces",
                "unexpand": "  unexpand [file]  — Convert spaces to tabs",
                "id": "  id  — Print user identity",
                "logname": "  logname  — Print login name",
                "mktemp": "  mktemp [-d]  — Create a temporary file or directory",
                "who": "  who  — Show who is logged on",
                "od": "  od [-x] <file>  — Dump file in octal/hex format",
                "join": "  join <file1> <file2>  — Join lines on common field",
                "history": "  history [n]  — Show command history (last n entries, default 20)",
                "fc": "  fc [-l] [n]  — List history, or re-run command by number (fc 42)",
                "alias": "  alias [name=cmd]  — List or set aliases",
                "unalias": "  unalias <name>  — Remove an alias",
                "export": "  export [NAME=VALUE]  — Set/show env vars (POSIX-style)",
                "set": '  set [name=value]  — Set/show env vars. $VAR, ${VAR}, and NAME=VALUE cmd supported',
                "source": "  source <file> | . <file>  — Execute commands from a file",
                "which": "  which <command>  — Locate a command",
                "type": "  type <command>  — Describe a command",

                "watch": "  watch <sec> <cmd>  — Run command repeatedly every N seconds",

                "procs": "  procs  — List running training jobs",
                "ps": "  ps  — List kernel processes (AI workloads)",
                "kill": "  kill <id>  — Stop a training job by ID",
                "train": "  train [dataset]  — Start training or list datasets",
                "bg": "  bg | jobs  — List background shell processes",
                "jobs": "  bg | jobs  — List background shell processes",
                "fg": "  fg <id>  — Bring a background process to foreground (wait for completion)",
                "models": "  models  — List available models (tab-completes names)",
                "load": "  load <name>  — Load a model (tab-completes names)",
                "unload": "  unload  — Unload the current model",
                "souls": "  souls  — List available souls (tab-completes names)",
                "switch": "  switch <name>  — Switch to a soul (tab-completes names)",
                "whoami": "  whoami  — Show current soul",
                "uptime": "  uptime  — How long Dait has been running",
                "health": "  health  — Quick API health check (colored status output)",
                "status": "  status  — Detailed system status (model, soul, server)",
                "metrics": "  metrics  — Show CPU/memory/disk metrics from server",
                "datasets": "  datasets  — List datasets (tab-completes names)",
                "knowledge": "  knowledge [query]  — List/search knowledge base entries",
                "remember": "  remember <fact>  — Store a fact in the knowledge base",
                "recall": "  recall <query>  — Search the knowledge base",
                "checkpoints": "  checkpoints  — List training checkpoints (tab-completes names)",
                "finetuned": "  finetuned  — List fine-tuned models (load <name> | rm <name>)",
                "protect": "  protect <model>  — Protect model files from accidental deletion (read-only + manifest)",
                "unprotect": "  unprotect <model>  — Remove protection from a model's files",
                "gen": "  gen <prompt>  — Generate text via inference",
                "tokenizer": "  tokenizer  — Show tokenizer vocabulary stats",
                "py": '  py <expr>  — Evaluate a Python expression. E.g. py 2 + 2, py [i*2 for i in range(5)]',


                "ai": '  ai <query>  — LLM-powered NL interpretation. E.g. ai "show me running jobs"',
                "agents": "  agents <goal>  — Multi-agent orchestration. E.g. agents 'research and write about X'",
                "tutorial": "  tutorial  — Interactive walkthrough of shell features",

                "remember": "  remember <fact>  — Store a fact in the knowledge base (also piped input)",
                "recall": "  recall <query>  — Search the knowledge base",
                "boot": "  boot  — Boot the shell (kernel + init services)",
                "shutdown": "  shutdown  — Halt all services and kernel",
                "svc": "  svc [list|start|stop|restart|status] [name]  — Manage init services",
                "devices": "  devices | lsdev  — List AI device nodes (/dev/*)",
                "lsdev": "  devices | lsdev  — List AI device nodes (/dev/*)",
                "asm": '  asm [file.asm] | asm --test | asm --list  — Assemble and run VM programs',
                "vmrun": '  vmrun [--admin|--kernel] [--steps=N] [--debug] <file|name>  — Run x86 assembly in virtual PC with RBAC. Built-in: hello, count, counter',
                "vmperms": "  vmperms  — Show x86 VM RBAC permission matrix (role×perm)",
                "permit": "  permit <cmd> [--persist]  — Grant permission for a blocked command",
                "deny": "  deny <cmd> [--persist]  — Revoke a previously granted permission",
                "permissions": "  permissions  — Show current permission policy and granted commands",
                "api": "  api [start|stop|status|restart]  — Manage the API server lifecycle",
                "chat": "  chat [msg] | chat /reset  — Multi-turn chat session",
                "confirm": "  confirm [on|off]  — Toggle auto-download confirmation",
                "events": "  events [filter] [n]  — Show recent EventBus events",
                "note": '  note [new|list|show|edit|delete|search|today|export]  — Development journal',
                "read": "  read [-p prompt] VARNAME  — Read stdin into a variable",
                "logs": '  logs [-l LEVEL] [-s SOURCE] [-n LINES] [-f] [--stats] [-e FILE] [--explain]  — Show/log panel. --explain: AI analysis of errors',
                "console": '  logs [-l LEVEL] [-s SOURCE] [-n LINES] [-f] [--stats] [-e FILE] [--explain]  — Same as "logs"',
                "tui": '  tui  — Launch three-pane TUI (console logs + shell output + input line)',
                "clear": "  clear  — Clear the terminal screen",
                "sleep": "  sleep <seconds>  — Sleep for N seconds (default 1)",
                "date": '  date [-u] [+format]  — Show current date and time',
                "cal": "  cal [[month] year]  — Show a calendar",
                "ln": "  ln [-s] <target> <link_name>  — Create hard or symbolic links",
                "render": "  render [sphere|cube|plane|light|mat|cam|go|neural|clear|preset]  — Path tracer + neural scene",
                "tui": '  tui  — Launch split-panel TUI (console + shell + input)',
            }
            if args in cmd_help:
                self._print(cmd_help[args])
            elif self.COMMANDS.get(args):
                fn = self.COMMANDS.get(args)
                doc = (fn.__doc__ or "").strip()
                if doc:
                    self._print(f"  {args}  — {doc.split(chr(10))[0]}")
                else:
                    self._print(f"  {args}  — (built-in command)")
            elif args in self._ext_cmds:
                h = getattr(self._ext_cmds[args], "help", "")
                if h and args not in h:
                    self._print(f"  {args}  — {h}")
                else:
                    self._print(f"  {args}  — (external command)")
            elif shutil.which(args):
                self._print(f"  {args}  — (system command)")
            elif args == "brief":
                self._print("""
Most common commands (help [cmd] for details, help for full list):
  help [cmd]           Show help
  exit / q / quit      Exit the shell
  history [n]          Show command history
  gen <prompt>         Generate text
  models               List models
  load <name>          Load model
  souls                List souls
  switch <name>        Switch soul
  whoami               Current soul
  health / status      System info
  datasets             List datasets
  remember <fact>      Store knowledge
  recall <query>       Search knowledge
  boot / shutdown      Shell lifecycle
  svc list             List services
  devices              List AI device nodes
  asm [file.asm]       Run VM program
  ai <query>           NL command interpretation
  procs                Show running jobs
  kill <id>            Stop a job
  permit / deny        Permission management
  permissions          Show permission policy
""")
            else:
                self._print(f"  Unknown command: {args}")
            return
        self._print(f"""
{_C_CYAN}Built-in commands:{_C_RESET}
  help [cmd]             Show this help or help for a specific command
  exit / q / quit         Exit the shell
  cd [dir]               Change directory (default: ~, - for previous)
  pwd                    Print working directory
  echo [text...]         Print text to stdout
  ls [dir]               List directory contents (VFS-aware)
  cat <file>             Print file contents (VFS-aware)
  mkdir <dir>            Create a directory
  rm [-rf] <path>        Remove files/directories
  touch <file>           Create empty file or update timestamp
  cp <src> <dst>         Copy files/directories
  mv <src> <dst>         Move or rename files
  head [-N] <file>       Output first N lines (VFS-aware)
  tail [-N] <file>       Output last N lines (VFS-aware)
  wc <file>              Count lines/words/chars (VFS-aware)
  grep [-i] <pattern>    Search for pattern in file or pipe (VFS-aware)
  sort [-rnu] [file]     Sort lines (supports pipes)
  uniq [file]            Remove adjacent duplicate lines (supports pipes)
  find [dir] -name <p>   Search for files by name pattern
  tee [-a] <file>        Copy stdin to file and stdout
  chmod <mode> <file>    Change file permissions (octal)
  du [-h] [path...]      Estimate disk usage
  diff <file1> <file2>   Compare files line by line
  stat <path>            Display file metadata
  nl [file]              Number lines of a file or piped input
  fold [-w w] [file]     Wrap long lines at a specified width
  tac [file]             Reverse lines (cat backwards)
  env / printenv         Print environment variables
  yes [string]           Repeatedly output a line
  realpath <path>        Resolve path to absolute
  dirname <path>         Strip last component from file path
  basename <path> [suf]  Strip directory from file path
  nproc                  Print number of CPUs
  hostname               Print system hostname
  uname [-a]             Print system information
  shuf [file]            Shuffle lines randomly (piped input)
  rev [file]             Reverse characters in each line
  paste <f1> [f2 ...]    Merge lines of files side by side
  comm <f1> <f2>         Compare two sorted files line by line
  test <expr>            Evaluate conditional expression ($? 0=true 1=false)
  printf <fmt> [args..]  Format and print data (%s %d %f \n \t)
  expand [file]          Convert tabs to spaces (piped input)
  unexpand [file]        Convert spaces to tabs (piped input)
  id                     Print user identity
  logname                Print login name
  mktemp [-d]            Create a temporary file or directory
  who                    Show who is logged on
  od [-x] <file>         Dump file in octal/hex format
  join <f1> <f2>         Join lines on a common field
  history [n]             Show command history
  fc [-l] [n]             List history, or re-run command #n (fc 42)
  alias [name=cmd]        List or set aliases
  unalias <name>          Remove an alias
  export [NAME=VALUE]     Set/show env vars (POSIX-style)
  set [name=value]        Set/show environment variables ($VAR expansion)
  source <file> / .       Execute commands from a file
  py <expr>               Evaluate a Python expression
  watch <sec> <cmd>       Run command repeatedly every N seconds

{_C_CYAN}Knowledge:{_C_RESET}
  remember <fact>         Store a fact in the knowledge base
  recall <query>          Search the knowledge base

{_C_CYAN}Scripting:{_C_RESET}
  which <cmd>             Locate a command
  type <cmd>              Describe a command

{_C_CYAN}Process management:{_C_RESET}
  procs / ps              List running training jobs
  kill <id>               Stop a training job
  train [dataset]         Start training (or list datasets)
  train status            Show training job status
  train follow <id>       Stream live training progress
  train stop <id>         Stop a training job
  train distill <ds>      Distill teacher into student
  train hf <model> <ds>   HuggingFace fine-tuning
  train auto [soul]       Auto-train with SloNet
  train load <cp>         Load a checkpoint
  train del <cp>          Delete a checkpoint
  bg / jobs               List background shell processes
  fg <id>                 Bring a background process to foreground

{_C_CYAN}Init system:{_C_RESET}
  boot                    Boot the shell (kernel + services)
  shutdown                Halt all services + kernel
  svc                     Service manager: list, start, stop, restart, status

{_C_CYAN}Devices:{_C_RESET}
  devices / lsdev         List AI device nodes (/dev/*)

{_C_CYAN}Model management:{_C_RESET}
  models                  List available models (tab-completes names)
  load <name>             Load a model (tab-completes names)
  unload                  Unload the current model

{_C_CYAN}Souls:{_C_RESET}
  souls                   List available souls (tab-completes names)
  switch <name>           Switch to a soul (tab-completes names)
  whoami                  Show current soul

{_C_CYAN}System:{_C_RESET}
  health                  Quick health check (colored status)
  status                  Detailed system status
  metrics                 CPU/memory/disk metrics
  uptime                  How long Dait has been running

{_C_CYAN}Data:{_C_RESET}
  datasets                List datasets (tab-completes names)
  knowledge               List knowledge base entries

{_C_CYAN}Training:{_C_RESET}
  checkpoints             List training checkpoints (tab-completes names)
  finetuned               List fine-tuned models (load <name> | rm <name>)

{_C_CYAN}Inference:{_C_RESET}
  gen <prompt>            Generate text
  tokenizer               Show tokenizer stats

{_C_CYAN}Shell features:{_C_RESET}
  <cmd> | <cmd>           Pipeline: output of first feeds second
  <cmd> &                 Background: run without blocking
  <cmd> && <cmd>          Chain: run next only if previous succeeded ($?=0)
  <cmd> || <cmd>          Chain: run next only if previous failed ($?!=0)
  <cmd> ; <cmd>           Chain: run next regardless of exit code
  <cmd> > <file>          Redirect output to file (overwrite)
  <cmd> > /dev/llm        Redirect output to AI device (write)
  <cmd> >> <file>         Redirect output to file (append)
  /dev/llm                AI device node: write prompt, read response
  /dev/null               Discard output: <cmd> > /dev/null
  /dev/random             Random tokens: cat /dev/random
  /dev/embedding          Compute embeddings: echo text > /dev/embedding
  /dev/knowledge          Knowledge base: read/write facts

{_C_CYAN}Permissions:{_C_RESET}
  permit <cmd>            Grant permission for a blocked command (this session)
  permit <cmd> --persist  Grant and save to disk (survives restart)
  permit --all-dangerous  Allow all dangerous commands at once
  deny <cmd>              Revoke a previously granted permission
  permissions             Show current policy (safe/elevated/dangerous/critical)

Virtual machine:
  asm [file.asm]          Assemble and run a VM program (.text + .data sections)
  asm --test              Run VM self-tests
  vmrun [--admin|--kernel] [--steps=N] [--debug] <file|name>
                          Run x86 assembly in X86VirtualSystem with RBAC
                          Built-in names: hello, count, counter
  vmrun --list            List available built-in x86 programs
  vmperms                 Show x86 VM RBAC permission matrix
  time <cmd>              Show command execution time
  $?                      Exit code of last command
  ai <query>              LLM-powered natural language interpretation
  agents <goal>           Multi-agent orchestration (researcher + writer + critic)
  $(cmd)                  Command substitution: inline output of cmd
  py <expr>               Evaluate Python expression
  $VAR / ${{VAR}}           Environment variable expansion
  NAME=VALUE cmd          Inline env var (set for single command)
  \\\\h \\\\w \\\\t \\\\u \\\\#    PS1 escapes: host, cwd, time, user, cmd#

{_C_CYAN}Development:{_C_RESET}
  note [new|list|show|edit|delete|search|today|export]  — Dev journal
  confirm [on|off]       Toggle auto-download confirmation
  render [sphere|cube|plane|light|mat|cam|go|neural|clear|preset]  — Path tracer
Examples:
  health
  models | head
  gen hello > output.txt
  time load gpt2
  ai show me running training jobs
  set PS1=$  ;  echo $HOME
  alias ll=procs
  source setup.sh
  py 2 + 2
"""[:-1])

    def _cmd_cd(self, args: str = "") -> None:
        """Change the working directory."""
        target = args.strip()
        if not target or target == "~":
            target = self._env.get("HOME", str(Path.home()))
        elif target == "-":
            target = self._env.get("OLDPWD", os.getcwd())
        try:
            old_cwd = os.getcwd()
            os.chdir(os.path.expanduser(target))
            self._env["OLDPWD"] = old_cwd
            self._last_exit_code = 0
        except FileNotFoundError:
            self._print(f"  cd: no such file or directory: {target}")
            self._last_exit_code = 1
        except PermissionError:
            self._print(f"  cd: permission denied: {target}")
            self._last_exit_code = 1
        except NotADirectoryError:
            self._print(f"  cd: not a directory: {target}")
            self._last_exit_code = 1

    def _cmd_pwd(self, args: str = "") -> None:
        """Print the working directory."""
        self._print(os.getcwd())
        self._last_exit_code = 0

    def _cmd_echo(self, args: str = "") -> None:
        """Echo arguments to stdout."""
        self._print(args)
        self._last_exit_code = 0

    def _cmd_ls(self, args: str = "") -> None:
        """List directory contents."""
        target = args.strip() or "."
        try:
            vfs = self.os.vfs
            if vfs and (target.startswith("/dev") or target.startswith("/proc")):
                entries = vfs.listdir(target)
            else:
                entries = os.listdir(os.path.expanduser(target))
            if entries is None:
                self._print(f"  ls: cannot access '{target}': No such file or directory")
                self._last_exit_code = 1
                return
            entries.sort()
            parts = []
            for e in entries:
                path = os.path.join(target, e) if target != "." else e
                if vfs:
                    is_dir = vfs.isdir(path)
                else:
                    is_dir = os.path.isdir(os.path.expanduser(path))
                suffix = "/" if is_dir else ""
                parts.append(e + suffix)
            if parts:
                self._print("  " + "  ".join(parts))
            self._last_exit_code = 0
        except FileNotFoundError:
            self._print(f"  ls: cannot access '{target}': No such file or directory")
            self._last_exit_code = 1
        except PermissionError:
            self._print(f"  ls: permission denied: {target}")
            self._last_exit_code = 1
        except NotADirectoryError:
            self._print(f"  ls: not a directory: {target}")
            self._last_exit_code = 1

    def _cmd_cat(self, args: str = "") -> None:
        """Concatenate and print files."""
        if not args:
            if self._piped_input:
                self._print(self._piped_input.rstrip("\n"))
                self._last_exit_code = 0
                return
            self._print("  Usage: cat <file>")
            self._last_exit_code = 1
            return
        target = args.strip()
        try:
            vfs = self.os.vfs
            if vfs and (target.startswith("/dev") or target.startswith("/proc")):
                content = vfs.read(target)
            else:
                content = Path(os.path.expanduser(target)).read_text()
            if content is None:
                self._print(f"  cat: {target}: No such file or directory")
                self._last_exit_code = 1
                return
            self._print(content.rstrip("\n"))
            self._last_exit_code = 0
        except FileNotFoundError:
            self._print(f"  cat: {target}: No such file or directory")
            self._last_exit_code = 1
        except IsADirectoryError:
            self._print(f"  cat: {target}: Is a directory")
            self._last_exit_code = 1
        except PermissionError:
            self._print(f"  cat: permission denied: {target}")
            self._last_exit_code = 1

    def _cmd_mkdir(self, args: str = "") -> None:
        """Create directories."""
        if not args:
            self._print("  Usage: mkdir <dir>")
            self._last_exit_code = 1
            return
        target = os.path.expanduser(args.strip())
        try:
            os.makedirs(target, exist_ok=False)
            self._last_exit_code = 0
        except FileExistsError:
            self._print(f"  mkdir: cannot create directory '{target}': File exists")
            self._last_exit_code = 1
        except PermissionError:
            self._print(f"  mkdir: permission denied: {target}")
            self._last_exit_code = 1
        except FileNotFoundError:
            self._print(f"  mkdir: cannot create directory '{target}': No such file or directory")
            self._last_exit_code = 1

    def _cmd_rm(self, args: str = "") -> None:
        """Remove files or directories."""
        if not args:
            self._print("  Usage: rm [-rf] <path>")
            self._last_exit_code = 1
            return
        parts = args.strip().split()
        flags = [p for p in parts if p.startswith("-")]
        paths = [p for p in parts if not p.startswith("-")]
        recursive = any(f in ("-r", "-rf", "-fr", "-R") for f in flags)
        force = any(f in ("-f", "-rf", "-fr") for f in flags)
        if not paths:
            self._print("  Usage: rm [-rf] <path>")
            self._last_exit_code = 1
            return
        for p in paths:
            target = os.path.expanduser(p)
            try:
                if os.path.isdir(target) and recursive:
                    import shutil as _shutil
                    _shutil.rmtree(target)
                elif os.path.isdir(target):
                    self._print(f"  rm: cannot remove '{p}': Is a directory")
                    self._last_exit_code = 1
                    continue
                else:
                    os.remove(target)
                self._last_exit_code = 0
            except FileNotFoundError:
                if not force:
                    self._print(f"  rm: cannot remove '{p}': No such file or directory")
                    self._last_exit_code = 1
            except PermissionError:
                self._print(f"  rm: permission denied: {p}")
                self._last_exit_code = 1

    def _cmd_touch(self, args: str = "") -> None:
        """Create empty files or update timestamps."""
        if not args:
            self._print("  Usage: touch <file> [file...]")
            self._last_exit_code = 1
            return
        for p in args.strip().split():
            target = os.path.expanduser(p)
            try:
                if os.path.exists(target):
                    os.utime(target, None)
                else:
                    Path(target).write_text("")
                self._last_exit_code = 0
            except PermissionError:
                self._print(f"  touch: permission denied: {p}")
                self._last_exit_code = 1

    def _cmd_cp(self, args: str = "") -> None:
        """Copy files."""
        if not args:
            self._print("  Usage: cp <src> <dst>")
            self._last_exit_code = 1
            return
        parts = args.strip().split()
        if len(parts) < 2:
            self._print("  cp: missing destination")
            self._last_exit_code = 1
            return
        src, dst = os.path.expanduser(parts[0]), os.path.expanduser(parts[1])
        try:
            import shutil as _shutil
            if os.path.isdir(src):
                _shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                _shutil.copy2(src, dst)
            self._last_exit_code = 0
        except FileNotFoundError:
            self._print(f"  cp: cannot stat '{parts[0]}': No such file or directory")
            self._last_exit_code = 1
        except PermissionError:
            self._print(f"  cp: permission denied")
            self._last_exit_code = 1

    def _cmd_mv(self, args: str = "") -> None:
        """Move or rename files."""
        if not args:
            self._print("  Usage: mv <src> <dst>")
            self._last_exit_code = 1
            return
        parts = args.strip().split()
        if len(parts) < 2:
            self._print("  mv: missing destination")
            self._last_exit_code = 1
            return
        src, dst = os.path.expanduser(parts[0]), os.path.expanduser(parts[1])
        try:
            os.rename(src, dst)
            self._last_exit_code = 0
        except FileNotFoundError:
            self._print(f"  mv: cannot stat '{parts[0]}': No such file or directory")
            self._last_exit_code = 1
        except PermissionError:
            self._print(f"  mv: permission denied")
            self._last_exit_code = 1

    def _cmd_head(self, args: str = "") -> None:
        """Output the first part of files (VFS-aware)."""
        parts = args.strip().split() if args else []
        n = 10
        targets = []
        for p in parts:
            if p.startswith("-") and p[1:].isdigit():
                n = int(p[1:])
            else:
                targets.append(p)
        # If no file args, use piped input
        if not targets:
            if self._piped_input:
                lines = self._piped_input.splitlines()
                self._print("\n".join(lines[:n]))
                self._last_exit_code = 0
                return
            self._print("  Usage: head [-N] <file>")
            self._last_exit_code = 1
            return
        for path in targets:
            target = os.path.expanduser(path)
            try:
                vfs = self.os.vfs
                if vfs and (target.startswith("/dev") or target.startswith("/proc")):
                    content = vfs.read(target)
                else:
                    content = Path(target).read_text()
                if content is None:
                    self._print(f"  head: {path}: No such file or directory")
                    self._last_exit_code = 1
                    continue
                lines = content.splitlines()
                out = "\n".join(lines[:n])
                if len(targets) > 1:
                    self._print(f"==> {path} <==")
                self._print(out)
                self._last_exit_code = 0
            except FileNotFoundError:
                self._print(f"  head: {path}: No such file or directory")
                self._last_exit_code = 1

    def _cmd_tail(self, args: str = "") -> None:
        """Output the last part of files (VFS-aware)."""
        parts = args.strip().split() if args else []
        n = 10
        targets = []
        for p in parts:
            if p.startswith("-") and p[1:].isdigit():
                n = int(p[1:])
            else:
                targets.append(p)
        # If no file args, use piped input
        if not targets:
            if self._piped_input:
                lines = self._piped_input.splitlines()
                self._print("\n".join(lines[-n:]))
                self._last_exit_code = 0
                return
            self._print("  Usage: tail [-N] <file>")
            self._last_exit_code = 1
            return
        for path in targets:
            target = os.path.expanduser(path)
            try:
                vfs = self.os.vfs
                if vfs and (target.startswith("/dev") or target.startswith("/proc")):
                    content = vfs.read(target)
                else:
                    content = Path(target).read_text()
                if content is None:
                    self._print(f"  tail: {path}: No such file or directory")
                    self._last_exit_code = 1
                    continue
                lines = content.splitlines()
                out = "\n".join(lines[-n:])
                if len(targets) > 1:
                    self._print(f"==> {path} <==")
                self._print(out)
                self._last_exit_code = 0
            except FileNotFoundError:
                self._print(f"  tail: {path}: No such file or directory")
                self._last_exit_code = 1

    def _cmd_wc(self, args: str = "") -> None:
        """Count lines, words, and characters (VFS-aware)."""
        if not args:
            if self._piped_input:
                lines = len(self._piped_input.splitlines())
                words = len(self._piped_input.split())
                chars = len(self._piped_input)
                self._print(f"  {lines:4} {words:4} {chars:4}")
                self._last_exit_code = 0
                return
            self._print("  Usage: wc <file>")
            self._last_exit_code = 1
            return
        target = os.path.expanduser(args.strip())
        try:
            vfs = self.os.vfs
            if vfs and (target.startswith("/dev") or target.startswith("/proc")):
                content = vfs.read(target)
            else:
                content = Path(target).read_text()
            if content is None:
                self._print(f"  wc: {args.strip()}: No such file or directory")
                self._last_exit_code = 1
                return
            lines = len(content.splitlines())
            words = len(content.split())
            chars = len(content)
            self._print(f"  {lines:4} {words:4} {chars:4} {args.strip()}")
            self._last_exit_code = 0
        except FileNotFoundError:
            self._print(f"  wc: {args.strip()}: No such file or directory")
            self._last_exit_code = 1

    def _cmd_grep(self, args: str = "") -> None:
        """Search for patterns in files or piped input (VFS-aware)."""
        if not args and not self._piped_input:
            self._print("  Usage: grep <pattern> [file]")
            self._last_exit_code = 1
            return
        import re as _re
        parts = args.strip().split()
        flags = [p for p in parts if p.startswith("-")]
        non_flags = [p for p in parts if not p.startswith("-")]
        ignore_case = any(f in ("-i", "-vi") for f in flags)
        invert = any(f in ("-v", "-vi") for f in flags)
        pattern = non_flags[0] if non_flags else ""
        target = non_flags[1] if len(non_flags) > 1 else None
        if not pattern:
            self._print("  Usage: grep <pattern> [file]")
            self._last_exit_code = 1
            return
        try:
            if target:
                target_path = os.path.expanduser(target)
                vfs = self.os.vfs
                if vfs and (target_path.startswith("/dev") or target_path.startswith("/proc")):
                    content = vfs.read(target_path)
                else:
                    content = Path(target_path).read_text()
                if content is None:
                    self._print(f"  grep: {target}: No such file or directory")
                    self._last_exit_code = 1
                    return
                lines = content.splitlines()
            else:
                lines = self._piped_input.splitlines()
            kwargs = {"flags": _re.IGNORECASE} if ignore_case else {}
            matched = 0
            for line in lines:
                found = _re.search(pattern, line, **kwargs) if kwargs else _re.search(pattern, line)
                if invert:
                    found = not found
                if found:
                    self._print(line)
                    matched += 1
            self._last_exit_code = 0 if matched else 1
        except _re.error as e:
            self._print(f"  grep: invalid pattern: {e}")
            self._last_exit_code = 2
        except FileNotFoundError:
            self._print(f"  grep: {target}: No such file or directory")
            self._last_exit_code = 1

    def _cmd_sort(self, args: str = "") -> None:
        """Sort lines of text (from file or piped input)."""
        parts = args.strip().split() if args else []
        flags = [p for p in parts if p.startswith("-")]
        targets = [p for p in parts if not p.startswith("-")]
        reverse = any(f in ("-r", "-R") for f in flags)
        numeric = any(f in ("-n", "-g") for f in flags)
        unique = any(f in ("-u",) for f in flags)
        if targets:
            target = os.path.expanduser(targets[0])
            try:
                lines = Path(target).read_text().splitlines()
            except FileNotFoundError:
                self._print(f"  sort: {targets[0]}: No such file or directory")
                self._last_exit_code = 1
                return
        elif self._piped_input:
            lines = self._piped_input.splitlines()
        else:
            self._print("  Usage: sort [-r] [-n] [-u] [file]")
            self._last_exit_code = 1
            return
        if numeric:
            lines.sort(key=lambda x: float(x.split()[0]) if x.split() else 0, reverse=reverse)
        else:
            lines.sort(reverse=reverse)
        if unique:
            seen = set()
            deduped = []
            for l in lines:
                if l not in seen:
                    seen.add(l)
                    deduped.append(l)
            lines = deduped
        self._print("\n".join(lines))
        self._last_exit_code = 0

    def _cmd_uniq(self, args: str = "") -> None:
        """Remove adjacent duplicate lines (from file or piped input)."""
        if args:
            target = os.path.expanduser(args.strip())
            try:
                lines = Path(target).read_text().splitlines()
            except FileNotFoundError:
                self._print(f"  uniq: {args.strip()}: No such file or directory")
                self._last_exit_code = 1
                return
        elif self._piped_input:
            lines = self._piped_input.splitlines()
        else:
            self._print("  Usage: uniq [file]")
            self._last_exit_code = 1
            return
        out = []
        prev = None
        for l in lines:
            if l != prev:
                out.append(l)
                prev = l
        self._print("\n".join(out))
        self._last_exit_code = 0

    def _cmd_find(self, args: str = "") -> None:
        """Search for files by name pattern (VFS-aware)."""
        if not args:
            self._print("  Usage: find [dir] -name <pattern>")
            self._last_exit_code = 1
            return
        parts = args.strip().split()
        search_dir = "."
        pattern = None
        i = 0
        while i < len(parts):
            if parts[i] in ("-name", "-iname") and i + 1 < len(parts):
                import fnmatch as _fnmatch
                pattern = parts[i + 1]
                if parts[i] == "-iname":
                    pattern = pattern.lower()
                    def _match_fn(name, pat=pattern):
                        return _fnmatch.fnmatch(name.lower(), pat)
                else:
                    _match_fn = lambda name, p=pattern: _fnmatch.fnmatch(name, p)
                i += 2
            elif not parts[i].startswith("-"):
                search_dir = parts[i]
                i += 1
            else:
                i += 1
        if pattern is None:
            self._print("  Usage: find [dir] -name <pattern>")
            self._last_exit_code = 1
            return
        search_path = os.path.expanduser(search_dir)
        try:
            matches = []
            for root, dirs, files in os.walk(search_path):
                for name in files + dirs:
                    if _match_fn(name):
                        matches.append(os.path.join(root, name))
            if matches:
                self._print("\n".join(matches))
            self._last_exit_code = 0
        except FileNotFoundError:
            self._print(f"  find: '{search_dir}': No such file or directory")
            self._last_exit_code = 1
        except PermissionError:
            self._print(f"  find: '{search_dir}': Permission denied")
            self._last_exit_code = 1

    def _cmd_tee(self, args: str = "") -> None:
        """Read stdin and write to both stdout and file(s)."""
        if not self._piped_input:
            self._print("  Usage: <command> | tee [-a] <file>")
            self._last_exit_code = 1
            return
        parts = args.strip().split() if args else []
        append = any(p == "-a" for p in parts)
        files = [p for p in parts if p != "-a"]
        mode = "a" if append else "w"
        for fname in files:
            try:
                with open(os.path.expanduser(fname), mode) as f:
                    f.write(self._piped_input)
                    if not self._piped_input.endswith("\n"):
                        f.write("\n")
            except (OSError, PermissionError) as e:
                self._print(f"  tee: {fname}: {e}")
                self._last_exit_code = 1
                return
        self._print(self._piped_input.rstrip("\n"))
        self._last_exit_code = 0

    def _cmd_xargs(self, args: str = "") -> None:
        """Build and execute command from stdin."""
        if not self._piped_input:
            self._print("  Usage: <command> | xargs [-n N] <cmd> [args...]")
            self._last_exit_code = 1
            return
        parts = args.strip().split()
        n = None
        cmd_parts = []
        i = 0
        while i < len(parts):
            if parts[i] == "-n" and i + 1 < len(parts):
                n = int(parts[i + 1])
                i += 2
            else:
                cmd_parts.append(parts[i])
                i += 1
        items = self._piped_input.split()
        if not cmd_parts:
            for item in items:
                self._print(item)
            self._last_exit_code = 0
            return
        if n:
            chunks = [items[i:i + n] for i in range(0, len(items), n)]
        else:
            chunks = [items]
        for chunk in chunks:
            full_cmd = cmd_parts + chunk
            if self._check_permission(full_cmd[0], " ".join(full_cmd[1:]) if len(full_cmd) > 1 else ""):
                result = self._execute_single(" ".join(full_cmd))
                if result:
                    self._print(result.rstrip("\n"))
        self._last_exit_code = 0

    def _cmd_time(self, args: str = "") -> None:
        """Time a command execution."""
        if not args:
            self._print("  Usage: time <command>")
            self._last_exit_code = 1
            return
        import time as _time
        start = _time.perf_counter()
        self._execute_single(args)
        elapsed = _time.perf_counter() - start
        self._print(f"  {_C_DIM}real  {elapsed:.3f}s{_C_RESET}")
        self._last_exit_code = 0

    def _cmd_chmod(self, args: str = "") -> None:
        """Change file permissions (chmod)."""
        if not args:
            self._print("  Usage: chmod <mode> <file>")
            self._last_exit_code = 1
            return
        parts = args.strip().split()
        if len(parts) < 2:
            self._print("  Usage: chmod <mode> <file>")
            self._last_exit_code = 1
            return
        mode, target = parts[0], os.path.expanduser(parts[1])
        try:
            if mode.isdigit():
                os.chmod(target, int(mode, 8))
            else:
                self._print(f"  chmod: symbolic modes not supported (use octal, e.g. 644)")
                self._last_exit_code = 1
                return
            self._last_exit_code = 0
        except FileNotFoundError:
            self._print(f"  chmod: cannot access '{parts[1]}': No such file or directory")
            self._last_exit_code = 1
        except PermissionError:
            self._print(f"  chmod: changing permissions of '{parts[1]}': Operation not permitted")
            self._last_exit_code = 1

    def _cmd_du(self, args: str = "") -> None:
        """Estimate disk usage of files/directories."""
        parts = args.strip().split() if args else []
        human = any(p == "-h" for p in parts)
        targets = [os.path.expanduser(p) for p in parts if p != "-h"]
        if not targets:
            targets = ["."]
        total = 0
        for target in targets:
            try:
                if os.path.isfile(target):
                    size = os.path.getsize(target)
                    total += size
                    label = self._format_size(size, human)
                    self._print(f"  {label}\t{target}")
                elif os.path.isdir(target):
                    sz = 0
                    for root, dirs, files in os.walk(target):
                        for f in files:
                            try:
                                sz += os.path.getsize(os.path.join(root, f))
                            except OSError:
                                pass
                    total += sz
                    label = self._format_size(sz, human)
                    self._print(f"  {label}\t{target}")
                else:
                    self._print(f"  du: cannot access '{target}': No such file or directory")
            except FileNotFoundError:
                self._print(f"  du: cannot access '{target}': No such file or directory")
        if len(targets) > 1:
            self._print(f"  {self._format_size(total, human)}\ttotal")
        self._last_exit_code = 0

    @staticmethod
    def _format_size(size: int, human: bool = False) -> str:
        if not human:
            return f"{size:>8}"
        for unit in ("B", "K", "M", "G", "T"):
            if size < 1024:
                return f"{size:>4.1f}{unit}"
            size /= 1024
        return f"{size:>4.1f}P"

    def _cmd_diff(self, args: str = "") -> None:
        """Compare two files line by line."""
        if not args:
            self._print("  Usage: diff <file1> <file2>")
            self._last_exit_code = 1
            return
        parts = args.strip().split()
        if len(parts) < 2:
            self._print("  Usage: diff <file1> <file2>")
            self._last_exit_code = 1
            return
        f1, f2 = os.path.expanduser(parts[0]), os.path.expanduser(parts[1])
        try:
            lines1 = Path(f1).read_text().splitlines()
            lines2 = Path(f2).read_text().splitlines()
        except FileNotFoundError as e:
            self._print(f"  diff: {e.filename}: No such file or directory")
            self._last_exit_code = 1
            return
        import difflib as _difflib
        differ = _difflib.Differ()
        diffs = list(differ.compare(lines1, lines2))
        changes = [l for l in diffs if l.startswith(("+ ", "- ", "? "))]
        if not changes:
            self._last_exit_code = 0
            return
        for l in diffs:
            if l.startswith("+ "):
                self._print(f"  {_C_GREEN}{l}{_C_RESET}")
            elif l.startswith("- "):
                self._print(f"  {_C_RED}{l}{_C_RESET}")
            elif l.startswith("? "):
                self._print(f"  {_C_DIM}{l}{_C_RESET}")
        self._last_exit_code = 1

    def _cmd_stat(self, args: str = "") -> None:
        """Display file or directory metadata."""
        if not args:
            self._print("  Usage: stat <path>")
            self._last_exit_code = 1
            return
        target = os.path.expanduser(args.strip())
        try:
            st = os.stat(target)
            import stat as _stat, time as _time
            mode_str = _stat.filemode(st.st_mode)
            size = st.st_size
            mtime = _time.strftime("%Y-%m-%d %H:%M:%S", _time.localtime(st.st_mtime))
            atime = _time.strftime("%Y-%m-%d %H:%M:%S", _time.localtime(st.st_atime))
            kind = "directory" if os.path.isdir(target) else "file" if os.path.isfile(target) else "other"
            self._print(f"  File: {target}")
            self._print(f"  Size: {size:,} bytes  {self._format_size(size, human=True).strip()}")
            self._print(f"  Type: {kind}")
            self._print(f"  Mode: {mode_str} ({oct(_stat.S_IMODE(st.st_mode))})")
            self._print(f"  Modified: {mtime}")
            self._print(f"  Accessed: {atime}")
            self._last_exit_code = 0
        except FileNotFoundError:
            self._print(f"  stat: cannot stat '{args.strip()}': No such file or directory")
            self._last_exit_code = 1
        except PermissionError:
            self._print(f"  stat: cannot stat '{args.strip()}': Permission denied")
            self._last_exit_code = 1

    def _cmd_cut(self, args: str = "") -> None:
        """Cut fields from lines of text (file or piped input)."""
        if not args and not self._piped_input:
            self._print("  Usage: cut -f<N> [-d<delim>] [file]")
            self._last_exit_code = 1
            return
        parts = args.strip().split() if args else []
        delim = "\t"
        fields = []
        target = None
        for p in parts:
            if p.startswith("-d") and len(p) > 2:
                delim = p[2:]
            elif p.startswith("-f") and len(p) > 2:
                for part in p[2:].split(","):
                    if "-" in part:
                        a, b = part.split("-", 1)
                        fields.extend(range(int(a) if a else 1, (int(b) if b else 9999) + 1))
                    else:
                        fields.append(int(part))
            elif not p.startswith("-"):
                target = p
        if not fields:
            self._print("  cut: you must specify a list of fields (-f)")
            self._last_exit_code = 1
            return
        try:
            if target:
                content = Path(os.path.expanduser(target)).read_text()
            elif self._piped_input:
                content = self._piped_input
            else:
                self._print("  cut: no input")
                self._last_exit_code = 1
                return
        except FileNotFoundError:
            self._print(f"  cut: {target}: No such file or directory")
            self._last_exit_code = 1
            return
        out_lines = []
        for line in content.splitlines():
            cols = line.split(delim)
            chosen = []
            for f in fields:
                if f <= len(cols):
                    chosen.append(cols[f - 1])
            out_lines.append(delim.join(chosen))
        self._print("\n".join(out_lines))
        self._last_exit_code = 0

    def _cmd_tr(self, args: str = "") -> None:
        """Translate or delete characters (piped input only)."""
        if not self._piped_input:
            self._print("  Usage: <command> | tr <set1> <set2>")
            self._last_exit_code = 1
            return
        parts = args.strip().split() if args else []
        delete = any(p == "-d" for p in parts)
        squeeze = any(p == "-s" for p in parts)
        sets = [p for p in parts if not p.startswith("-")]
        if len(sets) < 1 or (not delete and not squeeze and len(sets) < 2):
            self._print("  Usage: <command> | tr <set1> <set2>")
            self._last_exit_code = 1
            return
        set1 = sets[0]
        set2 = sets[1] if len(sets) > 1 else ""

        def _expand(s: str) -> str:
            result = []
            i = 0
            while i < len(s):
                if i + 2 < len(s) and s[i + 1] == "-" and ord(s[i]) < ord(s[i + 2]):
                    result.extend(chr(c) for c in range(ord(s[i]), ord(s[i + 2]) + 1))
                    i += 3
                else:
                    result.append(s[i])
                    i += 1
            return "".join(result)

        expanded1 = _expand(set1)
        expanded2 = _expand(set2)
        if delete:
            result = self._piped_input.translate(str.maketrans("", "", expanded1))
        elif squeeze:
            import re as _re
            result = _re.sub(rf"[{_re.escape(expanded1)}]+", lambda m: m.group(0)[0], self._piped_input)
        else:
            trans = str.maketrans(expanded1, expanded2[:len(expanded1)].ljust(len(expanded1), expanded2[-1] if expanded2 else ""))
            result = self._piped_input.translate(trans)
        self._print(result.rstrip("\n"))
        self._last_exit_code = 0

    def _cmd_seq(self, args: str = "") -> None:
        """Generate a sequence of numbers."""
        if not args:
            self._print("  Usage: seq [first [increment]] last")
            self._last_exit_code = 1
            return
        parts = args.strip().split()
        try:
            if len(parts) == 1:
                first, inc, last = 1, 1, float(parts[0])
            elif len(parts) == 2:
                first, inc, last = float(parts[0]), 1, float(parts[1])
            elif len(parts) == 3:
                first, inc, last = float(parts[0]), float(parts[1]), float(parts[2])
            else:
                self._print("  seq: too many arguments")
                self._last_exit_code = 1
                return
        except ValueError:
            self._print(f"  seq: invalid number")
            self._last_exit_code = 1
            return
        if first == int(first) and inc == int(inc) and last == int(last):
            fmt = "{:d}" if inc == int(inc) else "{:g}"
            nums = range(int(first), int(last) + 1, int(inc))
            self._print("\n".join(fmt.format(n) for n in nums))
        else:
            nums = []
            cur = first
            while cur <= last if inc > 0 else cur >= last:
                nums.append(str(cur))
                cur += inc
            self._print("\n".join(nums))
        self._last_exit_code = 0

    def _cmd_nl(self, args: str = "") -> None:
        """Number lines of a file or piped input."""
        if not args and not self._piped_input:
            self._print("  Usage: nl [file]")
            self._last_exit_code = 1
            return
        try:
            if args:
                content = Path(os.path.expanduser(args.strip())).read_text()
            else:
                content = self._piped_input
        except FileNotFoundError:
            self._print(f"  nl: {args.strip()}: No such file or directory")
            self._last_exit_code = 1
            return
        lines = content.splitlines()
        out = "\n".join(f"{i + 1}\t{line}" for i, line in enumerate(lines))
        self._print(out)
        self._last_exit_code = 0

    def _cmd_fold(self, args: str = "") -> None:
        """Wrap long lines at a specified width (default 80)."""
        if not args and not self._piped_input:
            self._print("  Usage: fold [-w width] [file]")
            self._last_exit_code = 1
            return
        parts = args.strip().split() if args else []
        width = 80
        target = None
        i = 0
        while i < len(parts):
            p = parts[i]
            if p == "-w" and i + 1 < len(parts):
                width = int(parts[i + 1])
                i += 2
            elif p.startswith("-w") and len(p) > 2:
                width = int(p[2:])
                i += 1
            elif p.startswith("-"):
                i += 1
            else:
                target = p
                i += 1
        try:
            if target:
                content = Path(os.path.expanduser(target)).read_text()
            elif self._piped_input:
                content = self._piped_input
            else:
                self._print("  fold: no input")
                self._last_exit_code = 1
                return
        except FileNotFoundError:
            self._print(f"  fold: {target}: No such file or directory")
            self._last_exit_code = 1
            return
        out_lines = []
        for line in content.splitlines():
            for i in range(0, len(line), width):
                out_lines.append(line[i:i + width])
        self._print("\n".join(out_lines))
        self._last_exit_code = 0

    def _cmd_tac(self, args: str = "") -> None:
        """Reverse lines of a file or piped input (cat backwards)."""
        if not args and not self._piped_input:
            self._print("  Usage: tac [file]")
            self._last_exit_code = 1
            return
        try:
            if args:
                content = Path(os.path.expanduser(args.strip())).read_text()
            else:
                content = self._piped_input
        except FileNotFoundError:
            self._print(f"  tac: {args.strip()}: No such file or directory")
            self._last_exit_code = 1
            return
        lines = content.splitlines()
        self._print("\n".join(reversed(lines)))
        self._last_exit_code = 0

    def _cmd_env(self, args: str = "") -> None:
        """Print environment variables."""
        for k, v in sorted(self._env.items()):
            self._print(f"  {k}={v}")
        self._last_exit_code = 0

    def _cmd_yes(self, args: str = "") -> None:
        """Repeatedly output a line (default: 'y')."""
        s = args.strip() or "y"
        for _ in range(100):
            self._print(s)
        self._last_exit_code = 0

    def _cmd_realpath(self, args: str = "") -> None:
        """Resolve path to absolute."""
        if not args:
            self._print("  Usage: realpath <path>")
            self._last_exit_code = 1
            return
        p = os.path.expanduser(args.strip())
        try:
            self._print(os.path.realpath(p))
            self._last_exit_code = 0
        except OSError as e:
            self._print(f"  realpath: {e}")
            self._last_exit_code = 1

    def _cmd_dirname(self, args: str = "") -> None:
        """Strip last component from file path."""
        if not args:
            self._print("  Usage: dirname <path>")
            self._last_exit_code = 1
            return
        self._print(os.path.dirname(os.path.expanduser(args.strip())))
        self._last_exit_code = 0

    def _cmd_basename(self, args: str = "") -> None:
        """Strip directory from file path."""
        if not args:
            self._print("  Usage: basename <path> [suffix]")
            self._last_exit_code = 1
            return
        parts = args.strip().split(None, 1)
        name = os.path.basename(os.path.expanduser(parts[0]))
        if len(parts) > 1 and name.endswith(parts[1]):
            name = name[:-len(parts[1])]
        self._print(name)
        self._last_exit_code = 0

    def _cmd_nproc(self, args: str = "") -> None:
        """Print number of CPUs."""
        import os as _os
        self._print(str(_os.cpu_count() or 1))
        self._last_exit_code = 0

    def _cmd_hostname(self, args: str = "") -> None:
        """Print system hostname."""
        import socket as _socket
        self._print(_socket.gethostname())
        self._last_exit_code = 0

    def _cmd_uname(self, args: str = "") -> None:
        """Print system information."""
        import platform as _platform
        flags = args.strip().split() if args else []
        if not flags or "-a" in flags:
            self._print(f"  {_platform.system()} {_platform.release()} {_platform.machine()}")
        else:
            parts = []
            for f in flags:
                if "s" in f:
                    parts.append(_platform.system())
                if "r" in f:
                    parts.append(_platform.release())
                if "m" in f:
                    parts.append(_platform.machine())
            self._print(" ".join(parts))
        self._last_exit_code = 0

    def _cmd_shuf(self, args: str = "") -> None:
        """Shuffle lines of a file or piped input."""
        if not args and not self._piped_input:
            self._print("  Usage: shuf [file]")
            self._last_exit_code = 1
            return
        try:
            if args:
                content = Path(os.path.expanduser(args.strip())).read_text()
            else:
                content = self._piped_input
        except FileNotFoundError:
            self._print(f"  shuf: {args.strip()}: No such file or directory")
            self._last_exit_code = 1
            return
        import random as _random
        lines = content.splitlines()
        _random.shuffle(lines)
        self._print("\n".join(lines))
        self._last_exit_code = 0

    def _cmd_rev(self, args: str = "") -> None:
        """Reverse characters in each line of a file or piped input."""
        if not args and not self._piped_input:
            self._print("  Usage: rev [file]")
            self._last_exit_code = 1
            return
        try:
            if args:
                content = Path(os.path.expanduser(args.strip())).read_text()
            else:
                content = self._piped_input
        except FileNotFoundError:
            self._print(f"  rev: {args.strip()}: No such file or directory")
            self._last_exit_code = 1
            return
        for line in content.splitlines():
            self._print(line[::-1])
        self._last_exit_code = 0

    def _cmd_paste(self, args: str = "") -> None:
        """Merge lines of files side by side."""
        if not args:
            self._print("  Usage: paste <file1> [file2 ...]")
            self._last_exit_code = 1
            return
        files = args.strip().split()
        try:
            readers = [Path(os.path.expanduser(f)).read_text().splitlines() for f in files]
        except FileNotFoundError as e:
            self._print(f"  paste: {e.filename}: No such file or directory")
            self._last_exit_code = 1
            return
        import itertools as _itertools
        for row in _itertools.zip_longest(*readers, fillvalue=""):
            self._print("\t".join(row))
        self._last_exit_code = 0

    def _cmd_comm(self, args: str = "") -> None:
        """Compare two sorted files line by line."""
        if not args:
            self._print("  Usage: comm <file1> <file2>")
            self._last_exit_code = 1
            return
        parts = args.strip().split()
        if len(parts) < 2:
            self._print("  Usage: comm <file1> <file2>")
            self._last_exit_code = 1
            return
        f1, f2 = os.path.expanduser(parts[0]), os.path.expanduser(parts[1])
        try:
            lines1 = Path(f1).read_text().splitlines()
            lines2 = Path(f2).read_text().splitlines()
        except FileNotFoundError as e:
            self._print(f"  comm: {e.filename}: No such file or directory")
            self._last_exit_code = 1
            return
        i = j = 0
        while i < len(lines1) and j < len(lines2):
            if lines1[i] < lines2[j]:
                self._print(f"\t\t{lines1[i]}")
                i += 1
            elif lines1[i] > lines2[j]:
                self._print(f"\t{lines2[j]}")
                j += 1
            else:
                self._print(lines1[i])
                i += 1
                j += 1
        while i < len(lines1):
            self._print(f"\t\t{lines1[i]}")
            i += 1
        while j < len(lines2):
            self._print(f"\t{lines2[j]}")
            j += 1
        self._last_exit_code = 0

    def _cmd_test(self, args: str = "") -> None:
        """Evaluate conditional expression. Sets exit code 0=true, 1=false."""
        if not args:
            self._last_exit_code = 1
            return
        parts = args.strip().split()
        if args.startswith("[ ") and args.endswith(" ]"):
            parts = args[2:-2].strip().split()
        # -f: file exists
        if len(parts) == 2 and parts[0] == "-f":
            self._last_exit_code = 0 if Path(os.path.expanduser(parts[1])).is_file() else 1
        elif len(parts) == 2 and parts[0] == "-d":
            self._last_exit_code = 0 if Path(os.path.expanduser(parts[1])).is_dir() else 1
        elif len(parts) == 2 and parts[0] == "-e":
            p = Path(os.path.expanduser(parts[1]))
            self._last_exit_code = 0 if p.exists() else 1
        elif len(parts) == 2 and parts[0] == "-z":
            self._last_exit_code = 0 if len(parts[1]) == 0 else 1
        elif len(parts) == 2 and parts[0] == "-n":
            self._last_exit_code = 0 if len(parts[1]) > 0 else 1
        elif len(parts) == 3 and parts[1] == "=":
            self._last_exit_code = 0 if parts[0] == parts[2] else 1
        elif len(parts) == 3 and parts[1] == "!=":
            self._last_exit_code = 0 if parts[0] != parts[2] else 1
        elif len(parts) == 3 and parts[1] == "-eq":
            self._last_exit_code = 0 if int(parts[0]) == int(parts[2]) else 1
        elif len(parts) == 3 and parts[1] == "-ne":
            self._last_exit_code = 0 if int(parts[0]) != int(parts[2]) else 1
        elif len(parts) == 3 and parts[1] == "-lt":
            self._last_exit_code = 0 if int(parts[0]) < int(parts[2]) else 1
        elif len(parts) == 3 and parts[1] == "-le":
            self._last_exit_code = 0 if int(parts[0]) <= int(parts[2]) else 1
        elif len(parts) == 3 and parts[1] == "-gt":
            self._last_exit_code = 0 if int(parts[0]) > int(parts[2]) else 1
        elif len(parts) == 3 and parts[1] == "-ge":
            self._last_exit_code = 0 if int(parts[0]) >= int(parts[2]) else 1
        else:
            self._last_exit_code = 1

    def _cmd_printf(self, args: str = "") -> None:
        """Format and print data (supports %s, %d, %f, \\n, \\t)."""
        if not args:
            self._last_exit_code = 1
            return
        parts = args.strip().split(maxsplit=1)
        fmt = parts[0]
        rest = parts[1] if len(parts) > 1 else ""
        fmt = fmt.replace("\\n", "\n").replace("\\t", "\t").replace("\\\\", "\\")
        # Handle %% format spec and count placeholders
        arg_parts = rest.split() if rest else []
        arg_idx = 0
        out = []
        i = 0
        while i < len(fmt):
            if fmt[i] == "%" and i + 1 < len(fmt):
                spec = fmt[i + 1]
                if spec == "%":
                    out.append("%")
                    i += 2
                elif spec == "s":
                    val = arg_parts[arg_idx] if arg_idx < len(arg_parts) else ""
                    arg_idx += 1
                    out.append(val)
                    i += 2
                elif spec == "d":
                    val = arg_parts[arg_idx] if arg_idx < len(arg_parts) else "0"
                    arg_idx += 1
                    try:
                        out.append(str(int(val)))
                    except ValueError:
                        out.append("0")
                    i += 2
                elif spec == "f":
                    val = arg_parts[arg_idx] if arg_idx < len(arg_parts) else "0.0"
                    arg_idx += 1
                    try:
                        out.append(f"{float(val):f}")
                    except ValueError:
                        out.append("0.000000")
                    i += 2
                else:
                    out.append(fmt[i])
                    i += 1
            else:
                out.append(fmt[i])
                i += 1
        self._print("".join(out).rstrip("\n"))
        self._last_exit_code = 0

    def _cmd_expand(self, args: str = "") -> None:
        """Convert tabs to spaces (piped input or file)."""
        if not args and not self._piped_input:
            self._print("  Usage: expand [file]")
            self._last_exit_code = 1
            return
        try:
            if args:
                content = Path(os.path.expanduser(args.strip())).read_text()
            else:
                content = self._piped_input
        except FileNotFoundError:
            self._print(f"  expand: {args.strip()}: No such file or directory")
            self._last_exit_code = 1
            return
        self._print(content.expandtabs(8).rstrip("\n"))
        self._last_exit_code = 0

    def _cmd_unexpand(self, args: str = "") -> None:
        """Convert spaces to tabs (piped input or file)."""
        if not args and not self._piped_input:
            self._print("  Usage: unexpand [file]")
            self._last_exit_code = 1
            return
        try:
            if args:
                content = Path(os.path.expanduser(args.strip())).read_text()
            else:
                content = self._piped_input
        except FileNotFoundError:
            self._print(f"  unexpand: {args.strip()}: No such file or directory")
            self._last_exit_code = 1
            return
        lines = content.splitlines()
        out = []
        for line in lines:
            spaces = 0
            for ch in line:
                if ch == " ":
                    spaces += 1
                else:
                    break
            tabs, rem = divmod(spaces, 8)
            out.append("\t" * tabs + " " * rem + line[spaces:])
        self._print("\n".join(out))
        self._last_exit_code = 0

    def _cmd_id(self, args: str = "") -> None:
        """Print user identity."""
        import getpass as _gp, os as _os
        user = _gp.getuser()
        uid = _os.getuid() if hasattr(_os, "getuid") else "?"
        gid = _os.getgid() if hasattr(_os, "getgid") else "?"
        self._print(f"  uid={uid}({user}) gid={gid}({user})")
        self._last_exit_code = 0

    def _cmd_logname(self, args: str = "") -> None:
        """Print login name."""
        import getpass as _gp, os as _os
        self._print(_gp.getuser())
        self._last_exit_code = 0

    def _cmd_mktemp(self, args: str = "") -> None:
        """Create a temporary file or directory."""
        import tempfile as _tf
        parts = args.strip().split()
        is_dir = any(p == "-d" for p in parts)
        try:
            if is_dir:
                path = _tf.mkdtemp()
            else:
                path = _tf.mkstemp()[1]
            self._print(path)
            self._last_exit_code = 0
        except OSError as e:
            self._print(f"  mktemp: {e}")
            self._last_exit_code = 1

    def _cmd_who(self, args: str = "") -> None:
        """Show who is logged on."""
        import os as _os, pwd as _pwd, time as _time
        try:
            host = _os.uname().nodename
        except AttributeError:
            host = "localhost"
        import getpass as _gp
        user = _gp.getuser()
        self._print(f"  {user}    console  {_time.strftime('%Y-%m-%d %H:%M')}")
        self._last_exit_code = 0

    def _cmd_od(self, args: str = "") -> None:
        """Dump file in octal/hex format."""
        if not args:
            self._print("  Usage: od <file>")
            self._last_exit_code = 1
            return
        parts = args.strip().split()
        target = None
        base = "o"
        for p in parts:
            if p == "-x":
                base = "x"
            elif p == "-o":
                base = "o"
            elif p == "-d":
                base = "d"
            elif not p.startswith("-"):
                target = p
        if not target:
            self._print("  od: no file specified")
            self._last_exit_code = 1
            return
        try:
            data = Path(os.path.expanduser(target)).read_bytes()
        except FileNotFoundError:
            self._print(f"  od: {target}: No such file or directory")
            self._last_exit_code = 1
            return
        for i in range(0, len(data), 16):
            chunk = data[i:i + 16]
            addr = f"{i:07o}" if base == "o" else f"{i:07x}"
            if base == "o":
                vals = " ".join(f"{b:03o}" for b in chunk)
            elif base == "x":
                vals = " ".join(f"{b:02x}" for b in chunk)
            else:
                vals = " ".join(f"{b:3d}" for b in chunk)
            ascii_repr = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
            self._print(f"  {addr} {vals:<48} {ascii_repr}")
        self._last_exit_code = 0

    def _cmd_join(self, args: str = "") -> None:
        """Join lines of two files on a common field."""
        if not args:
            self._print("  Usage: join <file1> <file2>")
            self._last_exit_code = 1
            return
        parts = args.strip().split()
        if len(parts) < 2:
            self._print("  Usage: join <file1> <file2>")
            self._last_exit_code = 1
            return
        f1, f2 = os.path.expanduser(parts[0]), os.path.expanduser(parts[1])
        try:
            lines1 = [l.split(None, 1) for l in Path(f1).read_text().splitlines()]
            lines2 = [l.split(None, 1) for l in Path(f2).read_text().splitlines()]
        except FileNotFoundError as e:
            self._print(f"  join: {e.filename}: No such file or directory")
            self._last_exit_code = 1
            return
        d1 = {l[0]: l[1] if len(l) > 1 else "" for l in lines1}
        d2 = {l[0]: l[1] if len(l) > 1 else "" for l in lines2}
        for key in sorted(set(d1) & set(d2)):
            self._print(f"{key} {d1[key]} {d2[key]}")
        self._last_exit_code = 0

    def _cmd_exit(self, args: str = "") -> None:
        self._running = False
        self._audit.shutdown()
        self.state.save()
        self.os.shutdown()
        self._print(f"  {_C_DIM}Shutting down...{_C_RESET}")

    def _cmd_history(self, args: str = "") -> None:
        n = 20
        if args and args.strip().isdigit():
            n = int(args.strip())
        lines = self._history[-n:] if n < len(self._history) else self._history
        start = max(1, len(self._history) - len(lines) + 1)
        for i, line in enumerate(lines, start):
            self._print(f"  {i:4d}  {line}")

    def _cmd_fc(self, args: str = "") -> None:
        """fc - list or re-run history commands (like bash fc)."""
        parts = args.strip().split()
        if not args:
            # No args: list all history (like `history`)
            self._cmd_history("")
            return
        # fc -l [n]: list last n commands
        if parts[0] == "-l":
            n = parts[1] if len(parts) > 1 else ""
            self._cmd_history(n)
            return
        # fc <n>: re-run command by history number
        try:
            n = int(parts[0])
        except ValueError:
            self._print(f"  Usage: fc [-l] [n]")
            self._print(f"    fc       — list history")
            self._print(f"    fc -l 5  — list last 5 commands")
            self._print(f"    fc 42    — re-run command #42")
            return
        if n < 1 or n > len(self._history):
            self._print(f"  No history entry #{n} (have {len(self._history)} entries)")
            return
        cmd = self._history[n - 1]
        self._print(f"  Re-running: {cmd}")
        cmds, is_bg, should_time = self._parse_pipeline(cmd)
        if is_bg:
            if len(cmds) > 1:
                self._execute_background_tuples(cmds)
            else:
                self._execute_background(cmd.rstrip("& ").strip())
        elif len(cmds) > 1:
            self._execute_pipeline(cmds, should_time=should_time)
        else:
            out = self._execute_single(cmd, "")
            self._print(out, end="")

    # ── Permission commands ────────────────────────────────────────

    def _cmd_permit(self, args: str = "") -> None:
        """permit <cmd> — grant permission for a command (session or persistent)."""
        from .permissions import Risk
        parts = args.strip().split()
        if not parts:
            self._print(f"  Usage: permit <cmd> [--persist]")
            self._print(f"         permit --all-<risk> [--persist]")
            self._print(f"  Risk levels: {Risk.SAFE}, {Risk.ELEVATED}, {Risk.DANGEROUS}, {Risk.CRITICAL}")
            granted = self._perms.list_granted()
            if granted:
                self._print(f"  Currently granted: {', '.join(granted)}")
            return
        persist = "--persist" in parts
        targets = [p for p in parts if p != "--persist"]
        for t in targets:
            if t.startswith("--all-"):
                risk = t[len("--all-"):]
                if risk not in (Risk.SAFE, Risk.ELEVATED, Risk.DANGEROUS, Risk.CRITICAL):
                    self._print(f"  Unknown risk level: {risk}")
                    continue
                self._perms.set_policy(risk, "allow")
                if persist:
                    self._perms._save_persistent()
                self._print(f"  All {risk} commands now allowed")
            else:
                self._perms.grant(t, persist=persist)
                self._print(f"  Granted: {t}" + (" (persistent)" if persist else ""))

    def _cmd_deny(self, args: str = "") -> None:
        """deny <cmd> — revoke permission for a command."""
        from .permissions import Risk
        parts = args.strip().split()
        if not parts:
            self._print(f"  Usage: deny <cmd> [--persist]")
            self._print(f"         deny --all-<risk> [--persist]")
            return
        persist = "--persist" in parts
        targets = [p for p in parts if p != "--persist"]
        for t in targets:
            if t.startswith("--all-"):
                risk = t[len("--all-"):]
                if risk not in (Risk.SAFE, Risk.ELEVATED, Risk.DANGEROUS, Risk.CRITICAL):
                    self._print(f"  Unknown risk level: {risk}")
                    continue
                self._perms.set_policy(risk, "deny")
                self._perms.revoke(t, persist=persist) if persist else None
                if persist:
                    self._perms._save_persistent()
                self._print(f"  All {risk} commands now denied")
            else:
                self._perms.revoke(t, persist=persist)
                self._print(f"  Revoked: {t}" + (" (persistent)" if persist else ""))

    def _cmd_permissions(self, args: str = "") -> None:
        """permissions — show current permission policy and granted commands."""
        from .permissions import Risk
        self._print(f"  Risk policies:")
        for risk in (Risk.SAFE, Risk.ELEVATED, Risk.DANGEROUS, Risk.CRITICAL):
            action = self._perms._policy.get(risk, "deny")
            icon = f"{_C_GREEN}✓ allow{_C_RESET}" if action == "allow" else f"{_C_RED}✗ deny{_C_RESET}"
            self._print(f"    {risk:10s} {icon}")
        granted = self._perms.list_granted()
        if granted:
            self._print(f"  Granted commands: {', '.join(granted)}")
        self._print(f"  Config: {self._perms._config_path}")

    def _cmd_confirm(self, args: str = "") -> None:
        """confirm [on|off] — toggle auto-download (skip download confirmations).

        Usage:
          confirm        Show current setting
          confirm on     Enable auto-download (persistent)
          confirm off    Disable auto-download (persistent)
        """
        import yaml
        arg = args.strip().lower()

        if not arg:
            # Show current setting
            try:
                from domains.infrastructure.config import get_config
                cfg = get_config()
                current = cfg.features.auto_download
            except Exception:
                current = os.environ.get("SLO_AUTO_DOWNLOAD", "") == "1"

            if current:
                self._print(f"  Auto-download: {_C_GREEN}ON{_C_RESET}")
                self._print(f"  Downloads will be confirmed automatically (no prompts)")
            else:
                self._print(f"  Auto-download: {_C_RED}OFF{_C_RESET}")
                self._print(f"  Downloads will require confirmation")
            self._print(f"  Toggle: confirm on / confirm off")
            return

        if arg not in ("on", "off", "yes", "no", "true", "false", "1", "0"):
            self._print(f"  Usage: confirm [on|off]")
            self._print(f"    on/yes/true/1  — enable auto-download")
            self._print(f"    off/no/false/0 — disable auto-download")
            return

        new_value = arg in ("on", "yes", "true", "1")

        # Update config file
        try:
            from domains.infrastructure.config import _REPO_ROOT, get_config
            config_path = _REPO_ROOT / "config" / "defaults.yaml"
            if config_path.exists():
                with open(config_path) as f:
                    config_data = yaml.safe_load(f) or {}
            else:
                config_data = {}

            if "features" not in config_data:
                config_data["features"] = {}
            config_data["features"]["auto_download"] = new_value

            with open(config_path, "w") as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

            # Reload config
            get_config().reload()

            if new_value:
                self._print(f"  Auto-download: {_C_GREEN}ON{_C_RESET}")
                self._print(f"  Downloads will be confirmed automatically")
            else:
                self._print(f"  Auto-download: {_C_RED}OFF{_C_RESET}")
                self._print(f"  Downloads will require confirmation")
            self._print(f"  Setting saved to {config_path}")

        except Exception as e:
            self._print(f"  Error updating config: {e}")
            self._print(f"  Fallback: export SLO_AUTO_DOWNLOAD=1")

    def _cmd_procs(self, args: str = "") -> None:
        jobs = self._spinner_call("Fetching jobs", lambda: self.cmds.ps(), ok_msg=None)
        if not jobs:
            self._print("  No running jobs")
            return
        rows = []
        for j in jobs:
            rows.append([
                str(j.get("id", ""))[:12],
                j.get("status", ""),
                str(j.get("name", "")),
                f"{j.get('progress', 0)}%",
                str(j.get("loss", "\u2014"))[:8],
            ])
        self._table(rows, ["ID", "Status", "Name", "Progress", "Loss"])

    def _cmd_ps(self, args: str = "") -> None:
        procs = self.os.kernel.list_processes()
        if not procs:
            self._print("  No kernel processes")
            return
        state_names = {
            0: "CREATED", 1: "READY", 2: "RUNNING",
            3: "WAITING", 4: "STOPPED", 5: "ZOMBIE",
        }
        rows = []
        for p in procs:
            state = state_names.get(p.state, str(p.state))
            age = time.time() - p.created_at
            rows.append([str(p.pid), p.name, state, f"{age:.1f}s"])
        self._table(rows, ["PID", "Name", "State", "Age"])

    def _cmd_kill(self, args: str = "") -> None:
        if not args:
            self._print("  Usage: kill <job_id>")
            return
        result = self._spinner_call("Killing job", lambda: self.cmds.kill(args.strip()), ok_msg=None)
        self._print(self._dump_json(result))

    def _cmd_load(self, args: str = "") -> None:
        if not args:
            self._print("  Usage: load <model_name>")
            return
        if not self._require_api("load"):
            return
        import sys
        import time
        model_name = args.strip()

        try:
            from domains.infrastructure.conversion_tracker import get_tracker
            from apps.cli.src.utils.progress import ProgressBar
            tracker = get_tracker()

            result_holder = [None]
            import threading

            def _load():
                result_holder[0] = self.cmds.load_model(model_name)

            t = threading.Thread(target=_load, daemon=True)
            t.start()

            bar = ProgressBar(total=100, desc=f"Loading {model_name}", width=30, show_eta=True)

            while t.is_alive():
                status = tracker.get(model_name)
                if status:
                    pct = int(status["progress"] * 100)
                    stage = status["stage"]

                    if stage in ("downloading", "converting", "protecting", "loading"):
                        bar.desc = status["message"][:40]
                        bar.set_progress(pct)
                    elif stage == "ready":
                        bar.set_progress(100)
                        break
                    elif stage == "error":
                        bar.finish()
                        self._print(f"  ✗ {status.get('error', 'Unknown error')}")
                        return

                time.sleep(0.15)

            t.join()
            bar.finish()

            result = result_holder[0]
            if result is None:
                self._print(f"  ✗ Load failed")
                return

            status = result.get("status", "?")
            if status == "loaded":
                self._print(f"  ✓ {model_name} loaded on {result.get('device', 'cpu')}")
            elif status == "error":
                self._print(f"  ✗ {result.get('error', 'Unknown error')}")

        except ImportError:
            self._print(f"  Loading {model_name}...")
            self._print("  (this may take 30-120s on CPU)")
            result = self.cmds.load_model(model_name)
            if result is None:
                self._print(f"  ✗ Load failed")
                return
            status = result.get("status", "?")
            if status == "loaded":
                self._print(f"  ✓ {model_name} loaded on {result.get('device', 'cpu')}")
            elif status == "error":
                self._print(f"  ✗ {result.get('error', 'Unknown error')}")

    def _cmd_uptime(self, args: str = "") -> None:
        """Print how long Dait has been running (like Unix uptime)."""
        uptime_secs = self.os.kernel.uptime if self.os.kernel else 0.0
        days = int(uptime_secs // 86400)
        hours = int((uptime_secs % 86400) // 3600)
        minutes = int((uptime_secs % 3600) // 60)
        if days > 0:
            self._print(f"  up {days} day{'s' if days > 1 else ''}, {hours}:{minutes:02d}")
        else:
            self._print(f"  up {hours}:{minutes:02d}")

    def _cmd_status(self, args: str = "") -> None:
        self._box(self.os.status_summary)
        try:
            detailed = self._spinner_call("Fetching status", lambda: self.cmds.health_detailed(), ok_msg=None)
            if isinstance(detailed, dict) and "registry" in detailed:
                registry = detailed.get("registry", {})
                models = registry.get("models", []) or registry.get("names", [])
                if models:
                    self._print(f"  Registry models: {len(models)}")
        except Exception:
            pass
        from .permissions import Risk
        granted = self._perms.list_granted()
        policies = []
        for risk in (Risk.SAFE, Risk.ELEVATED, Risk.DANGEROUS, Risk.CRITICAL):
            action = self._perms._policy.get(risk, "deny")
            policies.append(f"{risk}={action}")
        self._print(f"  Permissions: {', '.join(policies)}")
        if granted:
            self._print(f"  Granted: {', '.join(granted)}")

    def _cmd_events(self, args: str = "") -> None:
        """Show recent EventBus events. Optionally filter by event name and set limit.

        Usage:
          events              — show last 20 events
          events model         — filter by event names containing "model"
          events circuit 10    — filter by "circuit", show last 10
        """
        try:
            from domains.infrastructure.event_bus import get_event_bus
            bus = get_event_bus()
        except Exception:
            self._print("  EventBus not available")
            return

        parts = args.split()
        filter_event = parts[0] if parts else None
        limit = 20
        if len(parts) > 1:
            try:
                limit = int(parts[1])
            except ValueError:
                limit = 20

        all_events = bus.history()
        if not all_events:
            self._print("  No events recorded")
            return

        if filter_event:
            filtered = [e for e in all_events if filter_event.lower() in e.name.lower()]
        else:
            filtered = list(all_events)

        if not filtered:
            self._print(f"  No events matching '{filter_event}'")
            return

        events = filtered[-limit:]
        self._print(f"  Event history (last {len(events)} of {len(filtered)}, total {len(all_events)}):")
        for e in events:
            t = time.strftime("%H:%M:%S", time.localtime(e.timestamp))
            src = f"[{e.source}]" if e.source else ""
            data_str = str(e.data)[:80] if e.data else ""
            self._print(f"  {t}  {e.name:35s} {src:15s} {data_str}")

    def _cmd_metrics(self, args: str = "") -> None:
        metrics = self._spinner_call("Fetching metrics", lambda: self.cmds.system_metrics(), ok_msg=None)
        if metrics.get("error"):
            self._print(f"  Error: {metrics['error']}")
            return
        for k, v in metrics.items():
            if not k.startswith("_"):
                self._print(f"  {k}: {v}")

    def _cmd_tui(self, args: str = "") -> None:
        """Launch the split-panel TUI mode — console panel + shell output + input line."""
        try:
            from .tui_repl import TuiRepl
            tui = TuiRepl(self, self._log_buffer)
            tui.run()
        except ImportError as ex:
            self._print(f"  TUI mode not available: {ex}")
        except Exception as ex:
            self._print(f"  TUI error: {ex}")
            self._last_exit_code = 1

    def _cmd_logs(self, args: str = "") -> None:
        """Show the console log panel — infrastructure and API server logs.
        
        Flags:
          -l, --level LEVEL   filter by level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
          -s, --source SRC    filter by source substring
          -n, --lines N       show last N entries (default 30)
          -f, --follow        follow new entries (Ctrl+C to stop)
          -c, --clear         clear the buffer
          -e, --export FILE   save entries to a text file
              --stats         show log level distribution
              --explain       AI-powered analysis of recent errors/warnings
        """
        argv = args.split()
        level_filter = None
        source_filter = None
        count = 30
        follow = False
        export_path = None
        show_stats = False
        explain = False
        i = 0
        while i < len(argv):
            a = argv[i]
            if a in ("-l", "--level") and i + 1 < len(argv):
                level_filter = argv[i + 1].upper()
                i += 2
            elif a in ("-s", "--source") and i + 1 < len(argv):
                source_filter = argv[i + 1]
                i += 2
            elif a in ("-n", "--lines") and i + 1 < len(argv):
                try:
                    count = int(argv[i + 1])
                except ValueError:
                    pass
                i += 2
            elif a in ("-f", "--follow"):
                follow = True
                i += 1
            elif a in ("-c", "--clear"):
                self._log_buffer.clear()
                self._print("  Log buffer cleared.")
                return
            elif a in ("-e", "--export") and i + 1 < len(argv):
                export_path = argv[i + 1]
                i += 2
            elif a == "--stats":
                show_stats = True
                i += 1
            elif a == "--explain":
                explain = True
                i += 1
            else:
                i += 1

        if show_stats:
            all_entries = self._log_buffer.get()
            if not all_entries:
                self._print("  No log entries.")
                return
            from collections import Counter as _Counter
            levels = _Counter(e.level for e in all_entries)
            sources = _Counter(e.source for e in all_entries)
            total = len(all_entries)
            sep = f"  {_C_DIM}{'─' * 40}{_C_RESET}"
            self._print(f"  {_C_BOLD}Log Statistics{_C_RESET}  {_C_DIM}{total} total entries{_C_RESET}")
            self._print(sep)
            self._print(f"  {_C_BOLD}By Level:{_C_RESET}")
            for lvl in ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"):
                n = levels.get(lvl, 0)
                color = _C_RED if lvl in ("ERROR", "CRITICAL") else \
                        _C_YELLOW if lvl == "WARNING" else \
                        _C_GREEN if lvl == "INFO" else _C_CYAN
                bar = "█" * min(n, 40)
                self._print(f"    {color}{lvl:<9s}{_C_RESET} {n:>5d}  {_C_DIM}{bar}{_C_RESET}")
            self._print()
            self._print(f"  {_C_BOLD}Top Sources:{_C_RESET}")
            for src, n in sources.most_common(10):
                self._print(f"    {_C_DIM}{src:<35s}{_C_RESET} {n:>5d}")
            self._print()
            self._print(f"  {_C_BOLD}Time Range:{_C_RESET}")
            if total > 0:
                from datetime import datetime as _dt
                t0_raw = all_entries[0].timestamp
                t1_raw = all_entries[-1].timestamp
                t0 = _dt.fromtimestamp(t0_raw).strftime("%Y-%m-%d %H:%M:%S")
                t1 = _dt.fromtimestamp(t1_raw).strftime("%Y-%m-%d %H:%M:%S")
                span = t1_raw - t0_raw
                self._print(f"    {t0}  →  {t1}  ({_C_DIM}{span:.0f}s span{_C_RESET})")
            return

        if explain:
            err_entries = self._log_buffer.get(level=None, source=None, limit=50)
            err_entries = [e for e in err_entries if e.level in ("ERROR", "CRITICAL", "WARNING")]
            if not err_entries:
                self._print("  No errors or warnings to explain.")
                return
            status = self.os.api_status
            if not status.get("available"):
                self._print("  API server not available — cannot analyze logs.")
                return
            from datetime import datetime as _dt
            log_text = "\n".join(
                f"[{_dt.fromtimestamp(e.timestamp).strftime('%H:%M:%S')}] [{e.level}] [{e.source}] {e.message}"
                for e in err_entries[-20:]
            )
            prompt = (
                "You are a shell log analyzer. Given the following log entries, "
                "identify the most important issues, explain likely causes, "
                "and suggest fixes. Be concise (3-5 bullet points).\n\n"
                f"Logs:\n{log_text}\n\n"
                "Analysis:"
            )
            self._print(f"  {_C_BOLD}Log Analysis{_C_RESET} {_C_DIM}({len(err_entries)} errors/warnings){_C_RESET}")
            self._print(f"  {_C_DIM}{'─' * 40}{_C_RESET}")
            result = self._spinner_call("Analyzing", lambda: self.cmds.generate(prompt, max_tokens=200))
            if isinstance(result, dict) and "text" in result:
                analysis = result["text"].strip()
                for line in analysis.split("\n"):
                    self._print(f"  {line}")
            else:
                error = result.get("error", "unknown")
                self._print(f"  Analysis failed: {error}")
            return

        if export_path:
            entries = self._log_buffer.get(level=level_filter, source=source_filter)
            if not entries:
                self._print("  No log entries to export.")
                return
            try:
                from datetime import datetime as _dt
                with open(export_path, "w") as f:
                    for e in entries:
                        ts = _dt.fromtimestamp(e.timestamp).strftime("%Y-%m-%d %H:%M:%S")
                        f.write(f"[{ts}] [{e.level:<7s}] [{e.source}] {e.message}\n")
                self._print(f"  Exported {len(entries)} entries to {export_path}")
            except OSError as ex:
                self._print(f"  Error writing to {export_path}: {ex}")
            return

        def _render(entries):
            from datetime import datetime as _dt
            lines = []
            for e in entries:
                ts = _dt.fromtimestamp(e.timestamp).strftime("%H:%M:%S")
                color = _C_DIM
                if e.level == "ERROR" or e.level == "CRITICAL":
                    color = _C_RED
                elif e.level == "WARNING":
                    color = _C_YELLOW
                elif e.level == "INFO":
                    color = _C_GREEN
                elif e.level == "DEBUG":
                    color = _C_CYAN
                lines.append(f"  {_C_DIM}{ts}{_C_RESET} {color}{e.level:<7s}{_C_RESET} {_C_DIM}{e.source}{_C_RESET}  {e.message}")
            return lines

        entries = self._log_buffer.get(level=level_filter, source=source_filter, limit=count)
        if not entries:
            self._print("  No log entries.")
            return

        lines = _render(entries)
        sep = f"  {_C_DIM}{'─' * 40}{_C_RESET}"
        self._print(
            f"  {_C_BOLD}Console Logs{_C_RESET} {_C_DIM}({len(self._log_buffer)} buffered)"
            f"{'  -l ' + level_filter if level_filter else ''}"
            f"{'  -s ' + source_filter if source_filter else ''}"
            f"{_C_RESET}"
        )
        self._print(sep)
        for line in lines:
            self._print(line)
        self._print(sep)

        if follow:
            self._print(f"  {_C_DIM}Following — press Ctrl+C to stop{_C_RESET}")
            from datetime import datetime as _dt
            try:
                offset = len(self._log_buffer)
                while True:
                    time.sleep(0.5)
                    new_entries = self._log_buffer.get(
                        level=level_filter, source=source_filter, offset=offset
                    )
                    for e in new_entries:
                        ts = _dt.fromtimestamp(e.timestamp).strftime("%H:%M:%S")
                        color = _C_RED if e.level in ("ERROR", "CRITICAL") else \
                                _C_YELLOW if e.level == "WARNING" else \
                                _C_GREEN if e.level == "INFO" else _C_CYAN
                        self._print(f"  {_C_DIM}{ts}{_C_RESET} {color}{e.level:<7s}{_C_RESET} {_C_DIM}{e.source}{_C_RESET}  {e.message}")
                    offset += len(new_entries)
            except KeyboardInterrupt:
                self._print()

    def _cmd_protect(self, args: str = "") -> None:
        """Protect a model from accidental deletion: protect <model_id>"""
        model_id = args.strip()
        if not model_id:
            self._print("  Usage: protect <model_id>")
            self._print("  Makes model files read-only + drops .nomodeldelete marker")
            return
        try:
            from domains.infrastructure.model_protector import protect_model
            result = protect_model(model_id)
            n = len(result["protected"])
            errs = result["errors"]
            if n:
                self._print(f"  Protected {n} files for '{model_id}' (read-only + manifest)")
            else:
                self._print(f"  No files found to protect for '{model_id}'")
            if errs:
                for e in errs:
                    self._print(f"  Warning: {e['error']}")
        except Exception as e:
            self._print(f"  Error: {e}")

    def _cmd_unprotect(self, args: str = "") -> None:
        """Remove protection from a model: unprotect <model_id>"""
        model_id = args.strip()
        if not model_id:
            self._print("  Usage: unprotect <model_id>")
            return
        try:
            from domains.infrastructure.model_protector import unprotect_model
            result = unprotect_model(model_id)
            n = result["unprotected"]
            errs = result["errors"]
            if n:
                self._print(f"  Unprotected {n} files for '{model_id}'")
            else:
                self._print(f"  No protected files found for '{model_id}'")
            if errs:
                for e in errs:
                    self._print(f"  Warning: {e['error']}")
        except Exception as e:
            self._print(f"  Error: {e}")

    def _cmd_train(self, args: str = "") -> None:
        """Train: train [dataset] | train status | train follow <id> | train stop <id> | train distill <dataset> | train load-adapter <path> | train unload-adapter"""
        parts = args.strip().split()
        sub = parts[0] if parts else ""

        if not sub or sub in ("status", "follow", "stop", "distill", "hf", "auto", "load-adapter", "unload-adapter"):
            if not self._require_api("train"):
                return

        if sub == "status":
            jobs = self.cmds.train_status()
            if not jobs:
                self._print("  No training jobs")
                return
            rows = []
            for j in jobs:
                jid = j.get("id", "")[:8]
                status = j.get("status", "?")
                model = j.get("model", j.get("data_source", ""))
                prog = j.get("progress", 0)
                rows.append([jid, status, model, f"{prog}%"])
            self._table(rows, ["ID", "Status", "Model", "Progress"])
            return

        if sub == "follow":
            job_id = parts[1] if len(parts) > 1 else ""
            if not job_id:
                self._print("  Usage: train follow <job_id>")
                return
            self._stream_train_progress(job_id)
            return

        if sub == "stop":
            if len(parts) < 2:
                self._print("  Usage: train stop <job_id>")
                return
            r = self.cmds.train_stop(parts[1])
            self._print(f"  Stopped: {r}")
            return

        if sub == "distill":
            dataset = parts[1] if len(parts) > 1 else ""
            if not dataset:
                self._print("  Usage: train distill <dataset> [teacher] [epochs]")
                return
            teacher = parts[2] if len(parts) > 2 else "gpt2"
            epochs = int(parts[3]) if len(parts) > 3 else 5
            r = self._spinner_call("Starting distillation", lambda: self.cmds.train_distill(dataset, teacher=teacher, epochs=epochs))
            if "error" in r:
                self._print(f"  Error: {r['error']}")
            else:
                job_id = r.get("id", "")
                self._print(f"  Distillation started: {r.get('status', r)}")
                if job_id:
                    self._stream_train_progress(job_id)
            return

        if sub == "hf":
            model = parts[1] if len(parts) > 1 else ""
            dataset = parts[2] if len(parts) > 2 else ""
            if not model or not dataset:
                self._print("  Usage: train hf <model> <dataset> [epochs]")
                return
            epochs = int(parts[3]) if len(parts) > 3 else 3
            r = self._spinner_call("Starting fine-tune", lambda: self.cmds.train_hf(model, dataset, epochs=epochs))
            if "error" in r:
                self._print(f"  Error: {r['error']}")
            else:
                job_id = r.get("id", "")
                self._print(f"  Fine-tuning started: {r.get('status', r)}")
                if job_id:
                    self._stream_train_progress(job_id)
            return

        if sub == "load-adapter":
            path = parts[1] if len(parts) > 1 else ""
            if not path:
                self._print("  Usage: train load-adapter <adapter.npz> [--merge]")
                return
            merge = "--merge" in parts
            r = self._spinner_call("Loading adapter", lambda: self.cmds.load_adapter(path, merge=merge))
            if "error" in r:
                self._print(f"  Error: {r['error']}")
            else:
                rank = r.get("rank", "?")
                n_params = r.get("n_params", 0)
                merged = " (merged)" if r.get("merged") else ""
                self._print(f"  Loaded adapter: rank={rank}, {n_params:,} params{merged}")
            return

        if sub == "unload-adapter":
            r = self._spinner_call("Unloading adapter", lambda: self.cmds.unload_adapter())
            if "error" in r:
                self._print(f"  Error: {r['error']}")
            else:
                self._print(f"  {r.get('message', 'Adapter unloaded')}")
            return

        if sub == "auto":
            soul = parts[1] if len(parts) > 1 else ""
            teacher = parts[2] if len(parts) > 2 else "gpt2"
            epochs = int(parts[3]) if len(parts) > 3 else 10
            r = self._spinner_call("Starting auto-train", lambda: self.cmds.train_auto(soul_name=soul, teacher=teacher, epochs=epochs))
            if "error" in r:
                self._print(f"  Error: {r['error']}")
            else:
                self._print(f"  Auto-train started: {r.get('status', r)}")
            return

        if sub == "load":
            name = parts[1] if len(parts) > 1 else ""
            if not name:
                self._print("  Usage: train load <checkpoint_name>")
                return
            r = self.cmds.load_checkpoint(name)
            if "error" in r:
                self._print(f"  Error: {r['error']}")
            else:
                self._print(f"  Loaded: {name}")
            return

        if sub == "del":
            name = parts[1] if len(parts) > 1 else ""
            if not name:
                self._print("  Usage: train del <checkpoint_name>")
                return
            r = self.cmds.delete_checkpoint(name)
            if "error" in r:
                self._print(f"  Error: {r['error']}")
            else:
                self._print(f"  Deleted: {name}")
            return

        # Default: quick train on dataset (or list datasets if none specified)
        dataset = sub
        if not dataset:
            datasets = self.cmds.datasets()
            if not datasets:
                self._print("  No datasets available. Import data first.")
                return
            self._print("  Available datasets:")
            for d in datasets:
                self._print(f"    {d.get('name', d.get('id', '?'))}")
            self._print("\n  Usage: train <dataset>")
            return

        name = parts[1] if len(parts) > 1 else ""
        r = self._spinner_call("Starting training", lambda: self.cmds.train_quick(dataset, name=name))
        if "error" in r:
            self._print(f"  Error: {r['error']}")
        else:
            job_id = r.get("id", "")
            self._print(f"  Training started: {r.get('status', r)}")
            if job_id:
                self._stream_train_progress(job_id)

    def _stream_train_progress(self, job_id: str) -> None:
        """Stream training progress for a job with live progress bar."""
        import time
        from .commands import _api_get

        FILLED = "█"
        HALF = "▓"
        EMPTY = "░"
        bar_width = 32
        last_rendered = ""

        self._print(f"  Following job {job_id} (Ctrl+C to detach)")

        max_polls = 200
        for poll in range(max_polls):
            try:
                result = _api_get(f"/training/jobs/{job_id}")
                if not result:
                    self._print(f"  Job {job_id} not found")
                    return

                status = result.get("status", "unknown")
                if status == "unknown":
                    self._print(f"  Job {job_id} has no known status — detached")
                    return
                progress = result.get("progress", 0)
                epoch = result.get("current_epoch", result.get("epoch", 0))
                epochs = result.get("epochs", 0)
                loss = result.get("train_loss", result.get("loss", 0))

                # Build bar
                pct = progress / 100 if progress > 0 else 0
                filled = int(bar_width * pct)
                has_half = (bar_width * pct) - filled >= 0.5
                bar = FILLED * filled
                if has_half and filled < bar_width:
                    bar += HALF
                    bar += EMPTY * (bar_width - filled - 1)
                else:
                    bar += EMPTY * (bar_width - filled)

                line = f"  [{bar}] {progress:3d}%  epoch {epoch}/{epochs}  loss={loss or 0:.4f}  [{status}]"

                # In-place update using stdio
                if hasattr(self, '_stdio') and self._stdio:
                    self._stdio.progress(line, done=(status in ("completed", "failed", "error")))
                else:
                    # Fallback: manual in-place update with space-padding
                    pad = max(0, len(last_rendered) - len(line))
                    sys.stdout.write(f"\r{line}{' ' * pad}\r")
                    sys.stdout.flush()
                    last_rendered = line

                if status in ("completed", "failed", "error"):
                    if status == "completed":
                        ckpt = result.get("checkpoint", "")
                        self._print(f"\n  Training complete" + (f" — {ckpt}" if ckpt else ""))
                    else:
                        err = result.get("error", "unknown error")
                        self._print(f"\n  Training {status}: {err}")
                    return

            except KeyboardInterrupt:
                self._print("\n  Detached (job continues on server)")
                return
            except Exception as e:
                self._print(f"\n  Error: {e}")
                return

            time.sleep(3)

        self._print(f"  Job {job_id} still running after {max_polls} polls — detached")

    def _cmd_gen(self, args: str = "") -> None:
        if not args:
            self._print("  Usage: gen <prompt>")
            return
        if not self._require_api("gen"):
            return
        with self.console.spinner("Generating") as s:
            result = self.cmds.generate(args, max_tokens=150)
        if isinstance(result, dict) and "text" in result:
            self._print(f"\n  {result['text']}\n")
        elif isinstance(result, dict) and "error" in result:
            self._print(f"  Error: {result['error']}")
        else:
            self._print(self._dump_json(result))

    def _cmd_chat(self, args: str = "") -> None:
        """Multi-turn chat. Starts a session on first message. 'chat /reset' clears history."""
        if not args:
            self._print("  Usage: chat <message>")
            return
        if args == "/reset":
            self._chat_session_id = None
            self._chat_history = []
            self._print("  [session cleared]")
            return
        if not self._require_api("chat"):
            return
        if not self._chat_session_id:
            import uuid
            self._chat_session_id = str(uuid.uuid4())
            self._chat_history = []
            self._print("  [new session]")
        self._chat_history.append({"role": "user", "content": args})
        result = self._spinner_call("Thinking", lambda: self.cmds.chat(self._chat_history), "Done")
        if isinstance(result, dict) and "message" in result:
            text = result["message"]
            text = text.replace("<think>", "").replace("</think>", "")
            self._print(f"\n  {text.strip()}\n")
            self._chat_history.append({"role": "assistant", "content": text})
        elif isinstance(result, dict) and "error" in result:
            self._print(f"  Error: {result['error']}")
        else:
            self._print(self._dump_json(result))

    # ── LLM-powered NL interpreter ──────────────────────────────────

    def _cmd_render(self, args: str = "") -> None:
        """Path tracer + neural scene analysis. Subcommands:
  render                       — show scene info
  render sphere r x y z [mat]  — add sphere
  render cube  s x y z [mat]   — add cube
  render plane s y [mat]       — add plane
  render light x y z [r g b s] — add light
  render mat idx r g b m rough — set material
  render cam ox oy oz lx ly lz — set camera
  render go [w h spp]          — render image (prints stats)
  render neural                — render + neural analysis
  render clear                 — clear scene
  render preset <name>         — load preset scene (demo, Cornell, spheres)"""
        import numpy as _np
        from .cycles_device import CyclesDevice
        from .render_neural import RenderNeuralDevice

        if not hasattr(self, '_render_device'):
            self._render_device = CyclesDevice(width=80, height=60, samples=4)
            self._render_neural = RenderNeuralDevice(cycles_device=self._render_device)

        dev = self._render_device
        parts = args.strip().split()
        verb = parts[0].lower() if parts else ""

        def _f(s, default=0.0):
            try: return float(s)
            except (ValueError, TypeError): return default

        def _i(s, default=0):
            try: return int(s)
            except (ValueError, TypeError): return default

        if not verb or verb == "info":
            info = dev.call("info")
            self._print(f"  Scene: {info['meshes']} meshes, {info['materials']} materials, {info['lights']} lights")
            self._print(f"  Resolution: {info['resolution'][0]}x{info['resolution'][1]}, Samples: {info['samples']}")

        elif verb == "sphere":
            if len(parts) < 5:
                self._print("  Usage: render sphere radius cx cy cz [mat_idx]")
                return
            r, cx, cy, cz = _f(parts[1]), _f(parts[2]), _f(parts[3]), _f(parts[4])
            mat = _i(parts[5], 0) if len(parts) > 5 else 0
            idx = dev.call("add_sphere", r, cx, cy, cz, mat, 12)
            self._print(f"  Added sphere #{idx[0]}: r={r} center=({cx},{cy},{cz}) mat={mat}")

        elif verb == "cube":
            if len(parts) < 5:
                self._print("  Usage: render cube size cx cy cz [mat_idx]")
                return
            s, cx, cy, cz = _f(parts[1]), _f(parts[2]), _f(parts[3]), _f(parts[4])
            mat = _i(parts[5], 0) if len(parts) > 5 else 0
            idx = dev.call("add_cube", s, cx, cy, cz, mat)
            self._print(f"  Added cube #{idx[0]}: size={s} center=({cx},{cy},{cz}) mat={mat}")

        elif verb == "plane":
            if len(parts) < 3:
                self._print("  Usage: render plane size y [mat_idx]")
                return
            s, y = _f(parts[1]), _f(parts[2])
            mat = _i(parts[3], 0) if len(parts) > 3 else 0
            idx = dev.call("add_plane", s, y, mat)
            self._print(f"  Added plane #{idx[0]}: size={s} y={y} mat={mat}")

        elif verb == "light":
            if len(parts) < 4:
                self._print("  Usage: render light x y z [r g b strength]")
                return
            x, y, z = _f(parts[1]), _f(parts[2]), _f(parts[3])
            r = _f(parts[4], 1.0) if len(parts) > 4 else 1.0
            g = _f(parts[5], 1.0) if len(parts) > 5 else 1.0
            b = _f(parts[6], 1.0) if len(parts) > 6 else 1.0
            s = _f(parts[7], 5.0) if len(parts) > 7 else 5.0
            idx = dev.call("add_light", x, y, z, r, g, b, s)
            self._print(f"  Added light #{idx[0]}: ({x},{y},{z}) color=({r:.1f},{g:.1f},{b:.1f}) strength={s}")

        elif verb == "mat":
            if len(parts) < 7:
                self._print("  Usage: render mat idx r g b metallic roughness")
                return
            idx = _i(parts[1], 0)
            r, g, b = _f(parts[2]), _f(parts[3]), _f(parts[4])
            m, rough = _f(parts[5]), _f(parts[6])
            dev.call("set_material", idx, r, g, b, m, rough)
            self._print(f"  Material {idx}: color=({r:.2f},{g:.2f},{b:.2f}) metallic={m:.2f} roughness={rough:.2f}")

        elif verb == "cam":
            if len(parts) < 7:
                self._print("  Usage: render cam origin_x origin_y origin_z look_x look_y look_z [fov]")
                return
            ox, oy, oz = _f(parts[1]), _f(parts[2]), _f(parts[3])
            lx, ly, lz = _f(parts[4]), _f(parts[5]), _f(parts[6])
            fov = _f(parts[7], 50.0) if len(parts) > 7 else 50.0
            dev.call("set_camera", ox, oy, oz, lx, ly, lz, fov)
            self._print(f"  Camera: origin=({ox},{oy},{oz}) look_at=({lx},{ly},{lz}) fov={fov}")

        elif verb == "go":
            import time as _time
            w = _i(parts[1], 80) if len(parts) > 1 else 80
            h = _i(parts[2], 60) if len(parts) > 2 else 60
            spp = _i(parts[3], 4) if len(parts) > 3 else 4
            dev.call("set_resolution", w, h)
            dev.call("set_samples", spp)
            self._print(f"  Rendering {w}x{h} @ {spp} spp...")
            t0 = _time.time()
            img = dev.call("render")
            dt = _time.time() - t0
            nz = int((img.sum(axis=-1) > 0.01).sum())
            self._print(f"  Done in {dt:.1f}s — {nz}/{w*h} lit pixels ({100*nz/(w*h):.0f}%)")
            self._print(f"  Pixel range: [{img.min():.4f}, {img.max():.4f}]")

        elif verb == "neural":
            import time as _time
            w, h, spp = 80, 60, 4
            dev.call("set_resolution", w, h)
            dev.call("set_samples", spp)
            self._print(f"  Rendering {w}x{h} @ {spp} spp...")
            t0 = _time.time()
            out = self._render_neural.call("process")
            dt = _time.time() - t0
            desc = self._render_neural.call("descriptor")
            emb = out["embedding"]
            probs = out["probabilities"]
            cls_names = ["mat_unknown", "mat_diffuse", "mat_metallic", "mat_glass",
                         "mat_emissive", "mat_dielectric", "mat_rough", "mat_smooth"]
            dom = desc["dominant_class"]
            self._print(f"  Done in {dt:.1f}s")
            self._print(f"  Embedding: {emb.shape} (norm={_np.linalg.norm(emb):.4f})")
            self._print(f"  Dominant class: {cls_names[dom] if dom < len(cls_names) else dom}")
            self._print(f"  Class probs: {', '.join(f'{cls_names[i] if i < len(cls_names) else i}={p:.3f}' for i, p in enumerate(probs))}")
            self._print(f"  Entropy: {desc['neural_entropy']:.4f}")
            for k in ("image", "depth", "normal"):
                if k in desc:
                    self._print(f"  {k}: mean={desc[k]['mean']:.4f} std={desc[k]['std']:.4f}")

        elif verb == "clear":
            dev.call("clear")
            self._render_neural.call("set_source", dev)
            self._print("  Scene cleared.")

        elif verb == "preset":
            name = parts[1].lower() if len(parts) > 1 else ""
            self._apply_render_preset(name)
        else:
            self._print(f"  Unknown render subcommand: {verb}")
            self._print("  Try: render info | sphere | cube | plane | light | mat | cam | go | neural | clear | preset")

    def _apply_render_preset(self, name: str) -> None:
        """Load a preset scene configuration."""
        import numpy as _np
        dev = self._render_device

        if name == "demo":
            dev.call("clear")
            dev.call("set_material", 0, 0.3, 0.3, 0.35, 0.0, 0.8)
            dev.call("set_material", 1, 0.8, 0.1, 0.1, 0.1, 0.3)
            dev.call("set_material", 2, 0.9, 0.9, 0.9, 0.0, 0.0)
            dev.call("set_material", 3, 1.0, 1.0, 1.0, 0.0, 0.0)
            dev.call("add_plane", 6.0, -1.0, 0)
            dev.call("add_sphere", 0.6, -1.2, -0.4, 0.0, 1, 12)
            dev.call("add_sphere", 0.6, 0.0, -0.4, 0.0, 2, 12)
            dev.call("add_sphere", 0.6, 1.2, -0.4, 0.0, 1, 12)
            dev.call("add_cube", 0.4, 0.0, 1.5, 0.0, 3)
            dev.call("add_light", 2.0, 3.0, 2.0, 1.0, 0.95, 0.9, 8.0)
            dev.call("add_light", -2.0, 2.0, -1.0, 0.7, 0.8, 1.0, 4.0)
            dev.call("set_camera", 0, 1.5, 4, 0, 0, 0, 50)
            self._print("  Loaded preset: demo (3 spheres + cube + floor + 2 lights)")

        elif name == "cornell":
            dev.call("clear")
            dev.call("set_material", 0, 0.7, 0.1, 0.1, 0.0, 0.5)
            dev.call("set_material", 1, 0.1, 0.7, 0.1, 0.0, 0.5)
            dev.call("set_material", 2, 0.7, 0.7, 0.7, 0.0, 0.5)
            dev.call("set_material", 3, 1.0, 1.0, 1.0, 0.0, 0.0)
            dev.call("add_plane", 4.0, -1.0, 0)
            dev.call("add_cube", 1.0, -0.7, -0.5, 0.0, 0)
            dev.call("add_cube", 0.7, 0.7, -0.65, 0.0, 1)
            dev.call("add_cube", 0.3, 0.0, 1.5, 0.0, 3)
            dev.call("add_light", 0.0, 2.8, 0.0, 1.0, 0.95, 0.9, 10.0)
            dev.call("set_camera", 0, 1.0, 4.5, 0, 0.5, 0, 60)
            self._print("  Loaded preset: cornell (classic Cornell box)")

        elif name == "spheres":
            dev.call("clear")
            dev.call("set_material", 0, 0.3, 0.3, 0.35, 0.0, 0.8)
            for i in range(5):
                dev.call("set_material", i + 1, 0.5 + i * 0.1, 0.1, 0.1, float(i) / 5.0, 1.0 - float(i) / 5.0)
            dev.call("add_plane", 8.0, -1.0, 0)
            for i in range(5):
                dev.call("add_sphere", 0.5, -2.0 + i, -0.5, 0.0, i + 1, 12)
            dev.call("add_light", 0.0, 4.0, 2.0, 1.0, 0.95, 0.9, 8.0)
            dev.call("set_camera", 0, 1.5, 5, 0, 0, 0, 50)
            self._print("  Loaded preset: spheres (5 spheres, metallic→dielectric gradient)")

        else:
            self._print(f"  Unknown preset: {name}")
            self._print("  Available presets: demo, cornell, spheres")

    def _cmd_agents(self, args: str = "") -> None:
        """Multi-agent orchestration: agents <goal> or agents list."""
        from domains.agents.multi import get_orchestrator, SpecializedAgent
        orch = get_orchestrator()
        parts = args.strip().split(maxsplit=1)
        verb = parts[0].lower() if parts else ""

        if verb == "list":
            for a in orch.list_agents():
                self._print(f"  {a['name']:12s} — {a['role']}")
        elif verb in ("-h", "--help", "help"):
            self._print("  Usage:")
            self._print("    agents <goal>     — Run multi-agent on a goal")
            self._print("    agents list       — List available agents")
            self._print("    agents add <name> <role> <prompt> — Add agent (NYI)")
        elif verb in ("add",):
            rest = args.strip().split(maxsplit=1)[1] if len(args.strip().split()) > 1 else ""
            add_parts = rest.split(maxsplit=2)
            if len(add_parts) < 3:
                self._print("  Usage: agents add <name> <role> <system_prompt>")
                self._print("  Example: agents add summarizer summarize text into concise bullet points")
                return
            a_name, a_role, a_prompt = add_parts
            agent_key = a_name.lower().replace(" ", "_")
            agent = SpecializedAgent(
                name=a_name,
                role=a_role,
                system_prompt=a_prompt,
                tools=["memory"],
            )
            orch.agents[agent_key] = agent
            self._print(f"  \u2713 Agent '{a_name}' added (role: {a_role})")
            # Persist custom agents
            try:
                import json
                from pathlib import Path
                agents_file = Path.home() / ".config" / "sloughgpt" / "custom_agents.json"
                agents_file.parent.mkdir(parents=True, exist_ok=True)
                custom = {}
                if agents_file.exists():
                    custom = json.loads(agents_file.read_text())
                custom[agent_key] = {"name": a_name, "role": a_role, "system_prompt": a_prompt}
                agents_file.write_text(json.dumps(custom, indent=2))
            except Exception as e:
                self._print(f"  {_C_DIM}(not persisted: {e}){_C_RESET}")
        elif not args:
            self._cmd_help("agents")
        else:
            if not self._require_api("agents"):
                return
            goal = args.strip()
            self._print(f"  \U0001f916 Orchestrating agents for: {goal}")
            try:
                result = self._spinner_call("Planning & executing", lambda: orch.execute(goal))
                self._print("")
                self._print(result.get("response", "No response"))
                tasks = result.get("tasks", [])
                if tasks:
                    self._print(f"\n  {_C_DIM}Tasks: {len(tasks)}, "
                                f"completed: {sum(1 for t in tasks if t['status'] == 'completed')}"
                                f"{_C_RESET}")
            except Exception as e:
                self._print(f"  {_C_RED}Error:{_C_RESET} {e}")

    def _cmd_ai(self, args: str = "") -> None:
        if not args:
            self._print("  Usage: ai <natural language query>")
            self._print("  Example: ai show me running training jobs")
            return

        status = self.os.api_status
        if not status.get("available"):
            self._print("  \u2717 API server is not connected. Use \u2018api start\u2019 to launch it.")
            self._print("  Falling back to keyword matching...")
            self._interpret_natural(args)
            return

        available_commands = "\n".join(
            f"  {name} - {cmd.__doc__ or ''}"
            for name, cmd in sorted(self.COMMANDS.items())
        )
        for name, mod in sorted(self._ext_cmds.items()):
            h = getattr(mod, "help", "")
            available_commands += f"\n  {name} - {h}"

        # Build shell context
        ctx_parts = [f"  Current directory: {os.getcwd()}"]
        model = self._get_current_model()
        soul = self._get_current_soul()
        if model:
            ctx_parts.append(f"  Active model: {model}")
        if soul:
            ctx_parts.append(f"  Active soul: {soul}")
        recent = list(self._history)[-5:] if hasattr(self, "_history") else []
        if recent:
            ctx_parts.append(f"  Recent commands: {', '.join(recent)}")
        if self._log_buffer and len(self._log_buffer) > 0:
            err_entries = self._log_buffer.get(level="ERROR", limit=5)
            if err_entries:
                from datetime import datetime as _dt
                err_lines = []
                for e in err_entries:
                    ts = _dt.fromtimestamp(e.timestamp).strftime("%H:%M:%S")
                    err_lines.append(f"{ts} [{e.source}] {e.message}")
                ctx_parts.append(f"  Recent errors:\n" + "\n".join(f"    {l}" for l in err_lines))
        shell_context = "\n".join(ctx_parts)

        prompt = (
            "You are an AI shell assistant. Given the available commands and shell context below, "
            "interpret the user's natural language request and respond with ONLY "
            "the exact shell command to run. Do NOT include any explanation, "
            "backticks, or extra text. Just the command.\n\n"
            f"Shell context:\n{shell_context}\n\n"
            f"Available commands:\n{available_commands}\n\n"
            f"User request: {args}\n\n"
            "Command:"
        )

        self._print(f"  \u2601\ufe0f Interpreting as LLM query...")
        result = self._spinner_call("Thinking", lambda: self.cmds.generate(prompt, max_tokens=60))

        if isinstance(result, dict) and "text" in result:
            generated = result["text"].strip().split("\n")[0].strip()
            generated = generated.strip('`"\'')
            self._print(f"  \u2192 {generated}")
            self._print("")
            # Execute the generated command
            bg = generated.rstrip().endswith("&")
            cmds, _, _ = self._parse_pipeline(generated)
            if bg:
                if len(cmds) > 1:
                    self._execute_background_tuples(cmds)
                else:
                    self._execute_background(cmds[0][0])
            elif len(cmds) > 1:
                self._execute_pipeline(cmds)
            else:
                out = self._execute_single(cmds[0][0], "")
                self._print(out, end="")
        else:
            error = result.get("error", "unknown") if isinstance(result, dict) else "unexpected response"
            self._print(f"  AI interpretation failed: {error}")
            self._print("  Falling back to keyword matching...")
            self._interpret_natural(args)

    def _interpret_natural(self, query: str) -> None:
        """Keyword-based NL fallback when LLM is unavailable."""
        q = query.lower()
        if any(w in q for w in ["process", "job", "running", "ps", "procs"]):
            self._execute_single("procs")
        elif any(w in q for w in ["model", "models"]):
            self._print(self._execute_single("models"), end="")
        elif any(w in q for w in ["soul", "personality"]):
            self._print(self._execute_single("whoami"), end="")
        elif any(w in q for w in ["health", "status"]):
            self._print(self._execute_single("health"), end="")
        elif any(w in q for w in ["dataset", "data"]):
            self._print(self._execute_single("datasets"), end="")
        elif any(w in q for w in ["knowledge", "fact"]):
            self._print(self._execute_single("knowledge"), end="")
        elif any(w in q for w in ["checkpoint"]):
            self._print(self._execute_single("checkpoints"), end="")
        elif any(w in q for w in ["finetune", "trained"]):
            self._print(self._execute_single("finetuned"), end="")
        elif any(w in q for w in ["metric", "cpu", "memory", "disk"]):
            self._print(self._execute_single("metrics"), end="")
        elif any(w in q for w in ["tokenizer", "vocab"]):
            self._print(self._execute_single("tokenizer"), end="")
        elif any(w in q for w in ["help", "command"]):
            self._cmd_help()
        else:
            self._print(f"  Unknown query: {query}")
            self._print("  Try: ai show me running processes")

    def _show_welcome(self) -> None:
        """Show first-run welcome message."""
        terminal = shutil.get_terminal_size().columns
        w = self._print
        w("")
        w(f"  {_C_BOLD}{_C_CYAN}\u2728 Welcome to Dait{_C_RESET}")
        w(f"  {_C_DIM}{'─' * min(terminal, 50)}{_C_RESET}")
        w(f"  {_C_GREEN}Dait{_C_RESET} connects you to your local AI backend.")
        w(f"")
        w(f"  {_C_YELLOW}Quick start:{_C_RESET}")
        w(f"    health           Check server status")
        w(f"    models           List available models")
        w(f"    load gpt2        Load a model")
        w(f"    gen hello world  Generate text")
        w(f"    ai show models   Natural language commands")
        w(f"")
        w(f"  {_C_YELLOW}Pipes & more:{_C_RESET}")
        w(f"    health &         Run in background")
        w(f"    gen hello > out.txt  Redirect to file")
        w(f"    time load gpt2       Time a command")
        w(f"")
        w(f"  Type {_C_YELLOW}`tutorial`{_C_RESET} for an interactive walkthrough.")
        w(f"  Type {_C_YELLOW}`help`{_C_RESET} for all commands, {_C_YELLOW}`exit`{_C_RESET} to quit.")
        w(f"  {_C_DIM}{'─' * min(terminal, 50)}{_C_RESET}")
        w("")
        self.state.first_run = False
        self.state.save()

    def _cmd_tutorial(self, args: str = "") -> None:
        """Interactive walkthrough of shell features."""
        w = self._print
        _proceed = lambda: self.io.read(f"{_C_DIM}Press Enter to continue, or q to quit...{_C_RESET} ")

        steps = [
            ("\u2728 Welcome to the tutorial!", [
                "This will walk you through the shell features step by step.",
                "Press Enter to advance to each step, or 'q' to quit anytime.",
            ]),
            ("\u2460 Health check", [
                "  health  \u2014 Check if your AI server is running.",
                "  Try it: `health` shows API status, loaded model, and soul.",
            ]),
            ("\u2461 Models", [
                "  models  \u2014 List all available models.",
                "  load <name>  \u2014 Load a model (tab-complete names).",
                "  unload       \u2014 Unload the current model.",
                "  gen <prompt> \u2014 Generate text with the loaded model.",
            ]),
            ("\u2462 Souls & personality", [
                "  souls   \u2014 List available personality profiles.",
                "  switch <name>  \u2014 Switch to a soul.",
                "  whoami  \u2014 Show your current soul.",
            ]),
            ("\u2463 Pipelines", [
                "  Chain commands with |  (like bash):",
                "    models | head",
                "    gen hello > output.txt",
                "  Pipe to filters: head, tail, wc, sort, uniq",
                "  Sort flags: -r (reverse), -u (unique), -n (numeric)",
            ]),
            ("\u2464 Background & timing", [
                "  Append & to run in background:  health &",
                "  bg / jobs  \u2014 List background processes",
                "  fg <id>    \u2014 Wait for a background process",
                "  Prefix with `time` to measure:  time health",
            ]),
            ("\u2465 Redirection & env vars", [
                "  >  Redirect output to file:     gen hi > output.txt",
                "  >> Append to file:             gen hi >> output.txt",
                "  $VAR / ${VAR}  \u2014 Environment variables",
                "  NAME=VALUE cmd \u2014 Inline env for one command",
                "  set NAME=VALUE \u2014 Persistent env variables",
                "  $(cmd)        \u2014 Command substitution",
            ]),
            ("\u2466 Aliases & history", [
                "  alias ll=procs            \u2014 Create an alias",
                "  unalias <name>            \u2014 Remove an alias",
                "  history [n]               \u2014 Show command history",
                "  fc <n>                    \u2014 Re-run command #n",
                "  Ctrl+R / Ctrl+S           \u2014 Search history",
            ]),
            ("\u2467 PS1 & customization", [
                "  set PS1='\\\\u@\\\\h \\\\w $ '  \u2014 Custom prompt",
                "  \\\\h=host, \\\\w=cwd, \\\\t=time, \\\\u=user, \\\\s=shell, \\\\#=count",
                "  set NO_COLOR=1            \u2014 Disable colors",
                "  Source commands from file:  source setup.sh",
            ]),
            ("\u2468 AI mode & scripting", [
                "  ai <query>   \u2014 Natural language to commands",
                "    Example: ai show me running training jobs",
                "  py <expr>    \u2014 Evaluate Python inline",
                "    Example: py 2 + 2",
                "  Advanced: pipelines, watch, background jobs",
            ]),
            ("\u2469 All done!", [
                "  You're ready to use Dait!",
                "  Type `help` anytime for a full command reference.",
                "  Happy hacking! \U0001f680",
            ]),
        ]

        for title, lines in steps:
            w(f"\n  {_C_BOLD}{_C_CYAN}{title}{_C_RESET}")
            for line in lines:
                w(f"  {line}")
            try:
                resp = _proceed()
                if resp.strip().lower() == "q":
                    w(f"  {_C_DIM}Tutorial stopped.{_C_RESET}")
                    return
            except (EOFError, KeyboardInterrupt):
                w(f"  {_C_DIM}Tutorial stopped.{_C_RESET}")
                return

    def _cmd_lsdev(self, args: str = "") -> None:
        """List AI device nodes (/dev/*)."""
        if not self.os.devices:
            self._print("  Devices not available (not booted?)")
            return
        self._print("  AI Device nodes:")
        self._print(self.os.devices.list_devices())

    # ── Scripting commands ────────────────────────────────────────

    def _cmd_which(self, args: str = "") -> None:
        """Locate a command (like Unix which)."""
        if not args:
            self._print("  Usage: which <command>")
            self._last_exit_code = 1
            return
        cmd = args.strip().lower()
        if cmd in self.COMMANDS or cmd in self._ext_cmds or cmd in self._aliases:
            if cmd in self._aliases:
                self._print(f"  {cmd}: aliased to {self._aliases[cmd]}")
            elif cmd in self._ext_cmds:
                h = getattr(self._ext_cmds[cmd], "help", "")
                self._print(f"  {cmd}: external command — {h}")
            else:
                self._print(f"  {cmd}: shell built-in command")
            self._last_exit_code = 0
        else:
            found = shutil.which(cmd)
            if found:
                self._print(f"  {found}")
                self._last_exit_code = 0
            else:
                self._print(f"  {cmd}: not found")
                self._last_exit_code = 1

    def _cmd_type(self, args: str = "") -> None:
        """Describe a command (like Unix type)."""
        if not args:
            self._print("  Usage: type <command>")
            self._last_exit_code = 1
            return
        cmd = args.strip().lower()
        if cmd in self._aliases:
            self._print(f"  {cmd} is aliased to `{self._aliases[cmd]}`")
        elif cmd in self._ext_cmds:
            h = getattr(self._ext_cmds[cmd], "help", "")
            self._print(f"  {cmd} is an external command — {h}")
        elif cmd in self.COMMANDS:
            self._print(f"  {cmd} is a shell built-in")
        elif shutil.which(cmd):
            self._print(f"  {cmd} is {shutil.which(cmd)}")
        else:
            self._print(f"  {cmd}: not found")
            self._last_exit_code = 1

    def _cmd_read(self, args: str = "") -> None:
        """Read a line from stdin into a variable (like bash read).
        Usage: read [-p prompt] VARNAME"""
        prompt = ""
        parts = args.strip().split()
        if not parts:
            self._print("  Usage: read [-p prompt] VARNAME")
            self._last_exit_code = 1
            return
        if parts[0] == "-p" and len(parts) >= 2:
            prompt = parts[1] + " "
            parts = parts[2:]
        if not parts:
            self._print("  Usage: read [-p prompt] VARNAME")
            self._last_exit_code = 1
            return
        try:
            value = self.io.read(f"  {prompt}")
            self._env[parts[0]] = value
            self._last_exit_code = 0
        except (EOFError, KeyboardInterrupt):
            self._last_exit_code = 1

    # ── Init / Boot commands ────────────────────────────────────────


    def _cmd_api(self, args: str = "") -> None:
        """Manage the API server. Usage: api [start|stop|status|restart]"""
        parts = args.strip().split()
        cmd = parts[0] if parts else "status"
        api = self.os.api

        if cmd == "start":
            if api.is_running:
                self._status("info", "API server is already running.")
                return
            self._print("  Starting API server...")
            result = api.start()
            if result.get("ok"):
                self._status("ok", result.get('message', 'started'))
            else:
                self._status("error", result.get('error', 'failed to start'))
                self._last_exit_code = 1

        elif cmd == "stop":
            if not api.is_running:
                self._status("info", "API server is not running.")
                return
            self._print("  Stopping API server...")
            result = api.stop()
            self._status("ok", result.get('message', 'stopped'))

        elif cmd == "restart":
            if api.is_running:
                self._print("  Stopping API server...")
                api.stop()
            self._print("  Starting API server...")
            result = api.start()
            if result.get("ok"):
                self._status("ok", result.get('message', 'restarted'))
            else:
                self._status("error", result.get('error', 'failed to restart'))
                self._last_exit_code = 1

        else:  # status (default)
            status = api.status()
            if status.get("available"):
                model = status.get("model_id", "unknown") or "unknown"
                self._status("ok", f"API connected — {model} ({status.get('engine_type', '').strip() or 'cpu'})")
            else:
                self._status("error", "API not connected")
                self._print("  Use \u2018api start\u2019 to launch the API server.")
            if status.get("running"):
                uptime = status.get("uptime", 0)
                self._print(f"  Uptime: {uptime:.0f}s")

    def _require_api(self, cmd_name: str = "") -> bool:
        """Check API availability. Print warning and return False if down."""
        status = self.os.api_status
        if status.get("available"):
            return True
        self._print(f"  \u2717 API server is not connected. Use \u2018api start\u2019 to launch it.")
        self._last_exit_code = 1
        return False

    def _cmd_boot(self, args: str = "") -> None:
        """Boot the shell — start kernel + init system + services.

        Automatically starts the API server if not already running.
        """
        if self._running and self._piped_input is None:
            self._print("  Already booted. Use 'shutdown' to halt, then 'sloughgpt shell' to restart.")
            return
        self._running = True
        # Auto-start API if not available
        api = self.os.api
        if not api.is_running:
            self._print("  \u26a1 Auto-starting API server...")
            result = api.start()
            if not result.get("ok"):
                self._print(f"  \u2717 API auto-start failed: {result.get('error', 'unknown')}")
        result = self.os.boot(shell_run=self._shell_cmd if hasattr(self, "_shell_cmd") else None)
        if isinstance(result, tuple):
            log, api_status = result
        else:
            log, api_status = result, self.os.api_status
        if api_status.get("available"):
            model = api_status.get("model_id", "unknown") or "unknown"
            self._status("ok", f"API — {model}")
        else:
            self._status("error", "API not connected")

    def _cmd_shutdown(self, args: str = "") -> None:
        """Shut down the shell — halt all services + kernel."""
        log = self.os.shutdown()
        self._print(log)
        self._running = False

    def _cmd_svc(self, args: str = "") -> None:
        """Manage init services. Usage: svc [list|start|stop|restart|status] [name]"""
        if not self.os.init_system:
            self._print("  Init system not booted yet. Run 'boot' first.")
            self._last_exit_code = 1
            return

        parts = args.strip().split()
        cmd = parts[0] if parts else "list"
        name = parts[1] if len(parts) > 1 else ""

        init = self.os.init_system

        if cmd == "list" or cmd == "ls":
            self._print("  Services:")
            self._print(init.service_table())

        elif cmd == "status" or cmd == "st":
            if name:
                mgr = init.get_manager(name)
                if mgr:
                    self._print(mgr.status_line(len(name)))
                    for log_entry in mgr.instance.log[-5:]:
                        self._print(f"    {log_entry}")
                else:
                    self._print(f"  Unknown service: {name}")
                    self._last_exit_code = 1
            else:
                self._print("  Init status:")
                self._print(init.status_summary)

        elif cmd == "start":
            if not name:
                self._print("  Usage: svc start <name>")
                self._last_exit_code = 1
                return
            mgr = init.get_manager(name)
            if mgr:
                ok = mgr.start()
                self._print(f"  {name}: {'✓ started' if ok else '✗ failed'}")
            else:
                self._print(f"  Unknown service: {name}")
                self._last_exit_code = 1

        elif cmd == "stop":
            if not name:
                self._print("  Usage: svc stop <name>")
                self._last_exit_code = 1
                return
            mgr = init.get_manager(name)
            if mgr:
                mgr.stop()
                self._print(f"  {name}: stopped")
            else:
                self._print(f"  Unknown service: {name}")
                self._last_exit_code = 1

        elif cmd == "restart":
            if not name:
                self._print("  Usage: svc restart <name>")
                self._last_exit_code = 1
                return
            mgr = init.get_manager(name)
            if mgr:
                ok = mgr.restart()
                self._print(f"  {name}: {'✓ restarted' if ok else '✗ failed'}")
            else:
                self._print(f"  Unknown service: {name}")
                self._last_exit_code = 1

        elif cmd == "runlevel":
            self._print(f"  Current runlevel: {init.runlevel}")

        else:
            self._print("  Usage: svc [list|start|stop|restart|status] [name]")
            self._last_exit_code = 1

    def _cmd_asm(self, args: str = "") -> None:
        """Assemble and run a VM program. Usage: asm <file.asm>   or   piped | asm"""
        source = self._piped_input if self._piped_input else ""
        file_path = args.strip() if args else ""

        if file_path == "--test" or file_path == "--self-test":
            from domains.shell.vm import self_test as _vm_self_test
            results = _vm_self_test()
            self._print("  VM Self-Test:")
            for line in results:
                self._print(line)
            return

        if file_path == "--list" or file_path == "-l":
            self._print("  Built-in programs (use asm --test to run):")
            self._print("    hello      Hello World")
            self._print("    counter    Count 0..9")
            self._print("    fib        Fibonacci 0..12")
            self._print("    collatz    Collatz from 27")
            return

        if file_path:
            try:
                source = Path(os.path.expanduser(file_path)).read_text()
            except Exception as e:
                self._print(f"  asm: {e}")
                self._last_exit_code = 1
                return

        if not source:
            self._print("  Usage: asm <file.asm>   or   echo '<code>' | asm")
            self._print("         asm --test          Run self-tests")
            self._print("         asm --list          List built-in example programs")
            self._last_exit_code = 1
            return

        try:
            from domains.shell.vm import VMRunner, VMFault
            runner = VMRunner(devices=self.os.devices)
            output = runner.assemble_and_run(source)
            for line in output:
                self._print(line)
        except VMFault as e:
            self._print(f"  Assembly error: {e}")
            self._last_exit_code = 1
        except Exception as e:
            self._print(f"  VM error: {e}")
            self._last_exit_code = 1

    # ── x86 VM (X86VirtualSystem with RBAC) ────────────────────────

    def _cmd_vmperms(self, args: str = "") -> None:
        """Show x86 VM RBAC permission matrix."""
        from domains.shell.vm_permissions import Permission, Role, _ROLE_PERMISSIONS
        perms = list(Permission)
        roles = [Role.USER, Role.ADMIN, Role.KERNEL]
        col_w = max(len(p.name) for p in perms) + 2
        header = f"{'Permission':>{col_w}}  {'USER':>6} {'ADMIN':>6} {'KERNEL':>7}"
        self._print(header)
        self._print("─" * len(header))
        for p in perms:
            cells = " ".join("  ✓  " if p in _ROLE_PERMISSIONS[r] else "     " for r in roles)
            self._print(f"{p.name:>{col_w}}  {cells}")
        self._print("")

    # ── Built-in x86 assembly programs for vmrun ─────────────────────────

    HELLO_X86 = """\
[BITS 32]
mov eax, 3
mov ebx, 1
mov ecx, hello
mov edx, 18
int 0x80
mov eax, 1
xor ebx, ebx
int 0x80
jmp $
hello: db 'Hello from x86 VM!', 10
"""

    ECHO_X86 = """\
[BITS 32]
mov eax, 3
mov ebx, 1
mov ecx, msg
mov edx, 42
int 0x80
mov eax, 1
xor ebx, ebx
int 0x80
jmp $
msg: db 'echo: built-in x86 program (piped input not yet supported)', 10
"""

    FIB_X86 = """\
[BITS 32]
mov esi, 10
mov byte [num], '0'
.loop:
push esi
mov eax, 3
mov ebx, 1
mov ecx, num
mov edx, 1
int 0x80
mov eax, 3
mov ebx, 1
mov ecx, space
mov edx, 1
int 0x80
pop esi
inc byte [num]
dec esi
jnz .loop
mov eax, 3
mov ebx, 1
mov ecx, nl
mov edx, 1
int 0x80
mov eax, 1
xor ebx, ebx
int 0x80
num: db '0'
space: db ' '
nl: db 10
"""

    COLLATZ_X86 = """\
[BITS 32]
mov ecx, 5
mov byte [num], '0'
.loop:
push ecx
mov eax, 3
mov ebx, 1
mov ecx, num
mov edx, 1
int 0x80
mov eax, 3
mov ebx, 1
mov ecx, nl
mov edx, 1
int 0x80
pop ecx
inc byte [num]
dec ecx
jnz .loop
mov eax, 1
xor ebx, ebx
int 0x80
num: db '0'
nl: db 10
"""

    def _cmd_vmrun(self, args: str = "") -> None:
        """Run x86 assembly in X86VirtualSystem with RBAC.
        Usage: vmrun [--admin|--kernel] [--steps=N] [--debug] <file.asm>
               vmrun [--admin|--kernel] [--steps=N] [--debug] <name>   (built-in)
               echo '<code>' | vmrun [--admin|--kernel] [--steps=N] [--debug]
        Built-in names: hello, echo, fib, collatz
        """
        source = self._piped_input if self._piped_input else ""
        role = "user"
        max_steps = 5000
        debug = False
        rest = args.strip()

        # Parse flags
        while rest:
            if rest.startswith("--admin"):
                role = "admin"
                rest = rest[len("--admin"):].lstrip()
            elif rest.startswith("--kernel"):
                role = "kernel"
                rest = rest[len("--kernel"):].lstrip()
            elif rest.startswith("--steps="):
                try:
                    max_steps = int(rest[len("--steps="):].split()[0])
                    rest = rest[len("--steps=") + len(str(max_steps)):].lstrip()
                except ValueError:
                    self._print("  vmrun: --steps=N requires an integer")
                    self._last_exit_code = 1
                    return
            elif rest.startswith("--debug"):
                debug = True
                rest = rest[len("--debug"):].lstrip()
            elif rest.startswith("--list"):
                self._print("  Built-in x86 programs:")
                self._print(f"    {'hello':15s} Print 'Hello from x86 VM!'")
                self._print(f"    {'count':15s} Count 0 to 9")
                self._print(f"    {'counter':15s} Count 0 to 4")
                self._last_exit_code = 0
                return
            else:
                break

        max_role = os.environ.get("MAN_VM_ROLE", "kernel")
        max_idx = {"user": 0, "admin": 1, "kernel": 2}
        role_idx = {"user": 0, "admin": 1, "kernel": 2}
        if role_idx.get(role, 0) > max_idx.get(max_role, 2):
            self._print(f"  vmrun: --{role} requires MAN_VM_ROLE={role} or higher (current: {max_role})")
            self._last_exit_code = 1
            return

        file_or_name = rest if rest else ""

        # Check built-in programs by name
        builtins = {
            "hello": ShellREPL.HELLO_X86,
            "count": ShellREPL.FIB_X86,
            "counter": ShellREPL.COLLATZ_X86,
        }
        if file_or_name and file_or_name in builtins:
            source = builtins[file_or_name]
        elif file_or_name:
            try:
                source = Path(os.path.expanduser(file_or_name)).read_text()
            except Exception as e:
                self._print(f"  vmrun: {e}")
                self._last_exit_code = 1
                return

        if not source:
            self._print("  Usage: vmrun [--admin|--kernel] [--steps=N] [--debug] <file.asm>")
            self._print("         vmrun --list")
            self._print("         echo '<code>' | vmrun [--admin|--kernel]")
            self._last_exit_code = 1
            return

        try:
            from domains.shell.vm import X86VirtualSystem
            from domains.shell.vm_permissions import Role
            vs = X86VirtualSystem()

            pid = vs.spawn("user_prog", source)
            if pid is None:
                self._print("  vmrun: failed to spawn process")
                self._last_exit_code = 1
                return

            role_map = {"user": Role.USER, "admin": Role.ADMIN, "kernel": Role.KERNEL}
            vs._syscall._rbac.assign(pid, role_map[role])

            vs.scheduler.start(vs.cpu)
            current = vs.scheduler.current
            if current is None:
                self._print("  vmrun: no process to run")
                self._last_exit_code = 1
                return

            current.restore_to_cpu(vs.cpu)

            # Capture SYS_WRITE output instead of printing directly
            output_buffer: list[str] = []
            original_write = vs._syscall._sys_write

            def _captured_write(fd, buf_addr, count):
                if fd in (1, 2):
                    data = bytes(vs.cpu._read8(buf_addr + i) for i in range(count))
                    output_buffer.append(data.decode('ascii', errors='replace'))
                    return count
                return original_write(fd, buf_addr, count)

            vs._syscall._sys_write = _captured_write
            try:
                vs.cpu.run(max_steps=max_steps)
            finally:
                vs._syscall._sys_write = original_write

            exit_code = vs.cpu._regs[0] & 0xFFFFFFFF
            for line in output_buffer:
                self._print(line.rstrip())
            self._print(f"  [exit: {exit_code}, role: {role}, pid: {pid}, steps: {max_steps}]")

            if debug:
                reg_names = ["EAX", "ECX", "EDX", "EBX", "ESP", "EBP", "ESI", "EDI"]
                self._print(f"  Registers: {', '.join(f'{n}=0x{vs.cpu._regs[i]:08x}' for i, n in enumerate(reg_names))}")
                self._print(f"  EIP: 0x{vs.cpu._eip:08x}")

        except Exception as e:
            self._print(f"  vmrun error: {e}")
            self._last_exit_code = 1

    # ── Misc utility commands ────────────────────────────────────────

    def _cmd_clear(self, args: str = "") -> None:
        """Clear the terminal screen."""
        self._print("\033[2J\033[H", end="")

    def _cmd_sleep(self, args: str = "") -> None:
        """Sleep for N seconds: sleep <seconds>"""
        try:
            secs = float(args.strip())
        except ValueError:
            secs = 1.0
        import time as _time
        _time.sleep(secs)

    def _cmd_date(self, args: str = "") -> None:
        """Show current date and time: date [-u] [+format]
        -u: UTC time
        +format: strftime format (default: %a %b %d %H:%M:%S %Z %Y)"""
        from datetime import datetime as _dt, timezone as _tz
        argv = args.split()
        utc = False
        fmt = "%a %b %d %H:%M:%S %Z %Y"
        i = 0
        while i < len(argv):
            if argv[i] == "-u":
                utc = True
                i += 1
            elif argv[i].startswith("+"):
                fmt = argv[i][1:]
                i += 1
            else:
                i += 1
        now = _dt.now(_tz.utc if utc else None)
        self._print(now.strftime(fmt))

    def _cmd_cal(self, args: str = "") -> None:
        """Show a calendar: cal [[month] year]"""
        from datetime import datetime as _dt, timedelta as _td
        import calendar as _cal
        argv = args.split()
        now = _dt.now()
        if len(argv) == 0:
            year, month = now.year, now.month
        elif len(argv) == 1:
            year = int(argv[0])
            month = now.month if year == now.year else 1
        else:
            month, year = int(argv[0]), int(argv[1])
        if month < 1 or month > 12 or year < 1 or year > 9999:
            self._print(f"  cal: invalid date")
            return
        header = f"{_cal.month_name[month]} {year}".center(20)
        self._print(f"  {_C_BOLD}{header}{_C_RESET}")
        self._print(f"  Mo Tu We Th Fr Sa Su")
        first_dow = _cal.weekday(year, month, 1)
        days = _cal.monthrange(year, month)[1]
        line = "   " * first_dow
        for d in range(1, days + 1):
            line += f"{d:>2d} "
            if (first_dow + d) % 7 == 0:
                self._print(f"  {line}")
                line = ""
        if line.strip():
            self._print(f"  {line}")

    def _cmd_ln(self, args: str = "") -> None:
        """Create links: ln [-s] <target> <link_name>"""
        import shlex as _shlex
        argv = _shlex.split(args)
        symlink = False
        target = None
        link_name = None
        i = 0
        while i < len(argv):
            a = argv[i]
            if a in ("-s", "--symbolic"):
                symlink = True
                i += 1
            elif target is None:
                target = a
                i += 1
            elif link_name is None:
                link_name = a
                i += 1
            else:
                i += 1
        if not target or not link_name:
            self._print("  Usage: ln [-s] <target> <link_name>")
            self._last_exit_code = 1
            return
        import os as _os
        try:
            if symlink:
                _os.symlink(target, link_name)
            else:
                _os.link(target, link_name)
        except OSError as ex:
            self._print(f"  ln: {ex}")
            self._last_exit_code = 1

    # ── Notes (development journal) ────────────────────────────────

    def _cmd_note(self, args: str = "") -> None:
        """Development journal: note new/list/show/edit/delete/search/today/export."""
        from notes import get_note_store
        store = get_note_store(backend="mogdb")
        parts = args.split(None, 1)
        sub = parts[0].lower() if parts else "list"
        rest = parts[1] if len(parts) > 1 else ""

        if sub == "new":
            self._note_new(store, rest)
        elif sub == "list":
            self._note_list(store, rest)
        elif sub == "show":
            self._note_show(store, rest)
        elif sub == "edit":
            self._note_edit(store, rest)
        elif sub == "delete" or sub == "rm":
            self._note_delete(store, rest)
        elif sub == "search":
            self._note_search(store, rest)
        elif sub == "today":
            self._note_today(store)
        elif sub == "export":
            self._note_export(store, rest)
        elif sub == "tags":
            self._note_tags(store)
        elif sub == "status":
            self._note_status_summary(store)
        elif sub == "sprint":
            self._note_sprint(store, rest)
        elif sub == "timeline":
            self._note_timeline(store, rest)
        else:
            self._print(f"  note: unknown subcommand '{sub}'")
            self._print("  Usage: note <new|list|show|edit|delete|search|today|export|tags|status|sprint|timeline> [args]")
            self._last_exit_code = 1

    def _note_new(self, store, rest: str) -> None:
        if not rest:
            self._print("  Usage: note new <title> [--tags tag1,tag2] [--status s] [--sprint S1] [--gh owner/repo#123]")
            self._last_exit_code = 1
            return

        title = rest
        tags: list[str] = []
        status = "open"
        sprint = ""
        gh = ""

        for flag, handler in [
            ("--tags", lambda v: [t.strip() for t in v.split(",") if t.strip()]),
            ("--status", lambda v: v),
            ("--sprint", lambda v: v),
            ("--gh", lambda v: v),
        ]:
            if flag in title:
                idx = title.index(flag)
                before = title[:idx].strip()
                after = title[idx + len(flag) + 1:].strip()
                rest_val = after.split()[0] if after else ""
                if rest_val:
                    remainder = after[len(rest_val):].strip()
                    title = before + " " + remainder
                    if flag == "--tags":
                        tags = handler(rest_val)
                    elif flag == "--status":
                        status = handler(rest_val)
                    elif flag == "--sprint":
                        sprint = handler(rest_val)
                    elif flag == "--gh":
                        gh = handler(rest_val)
                title = title.strip()

        title = title.strip()
        if not title:
            self._print("  Title cannot be empty")
            self._last_exit_code = 1
            return

        note = store.create(title, tags=tags, status=status, sprint=sprint, gh=gh)
        sprint_tag = f" [{sprint}]" if sprint else ""
        self._print(f"  Created: {note.short_id}  {note.title}{sprint_tag}")
        self._last_exit_code = 0

    def _note_list(self, store, rest: str) -> None:
        tag = None
        status = None
        sprint = None
        limit = 20

        if "--tag" in rest:
            idx = rest.index("--tag") + 5
            tag = rest[idx:].split()[0] if rest[idx:].strip() else None
        if "--status" in rest:
            idx = rest.index("--status") + 8
            status = rest[idx:].split()[0] if rest[idx:].strip() else None
        if "--sprint" in rest:
            idx = rest.index("--sprint") + 8
            sprint = rest[idx:].split()[0] if rest[idx:].strip() else None
        if "--limit" in rest:
            idx = rest.index("--limit") + 7
            try:
                limit = int(rest[idx:].split()[0])
            except (ValueError, IndexError):
                pass

        notes = store.list_notes(tag=tag, status=status, sprint=sprint, limit=limit)
        if not notes:
            self._print("  No notes found.")
            return

        # Group by date
        by_date: dict[str, list] = {}
        for n in notes:
            by_date.setdefault(n.date_str, []).append(n)

        for date_str, day_notes in by_date.items():
            self._print(f"\n  {date_str}")
            for n in day_notes:
                tags_str = f"  [{', '.join(n.tags)}]" if n.tags else ""
                status_icon = {"open": "○", "wip": "◐", "done": "●", "blocked": "✕"}.get(n.status, "?")
                self._print(f"    {status_icon} {n.short_id}  {n.title}{tags_str}")
        self._print(f"\n  {len(notes)} note(s)")
        self._last_exit_code = 0

    def _note_show(self, store, rest: str) -> None:
        if not rest.strip():
            self._print("  Usage: note show <note-id>")
            self._last_exit_code = 1
            return

        note = store.get(rest.strip())
        if note is None:
            self._print(f"  Note not found: {rest.strip()}")
            self._last_exit_code = 1
            return

        tags_str = ", ".join(note.tags) if note.tags else "none"
        self._print(f"  {note.title}")
        self._print(f"  id: {note.id}")
        self._print(f"  created: {note.created_at}")
        self._print(f"  updated: {note.updated_at}")
        self._print(f"  status: {note.status}")
        self._print(f"  tags: {tags_str}")
        if note.sprint:
            self._print(f"  sprint: {note.sprint}")
        if note.gh:
            self._print(f"  gh: {note.gh}")
            if note.gh_url:
                self._print(f"  gh_url: {note.gh_url}")
        self._print("")
        for line in note.body.split("\n"):
            self._print(f"  {line}")
        self._last_exit_code = 0

    def _note_edit(self, store, rest: str) -> None:
        parts = rest.split(None, 1)
        if not parts:
            self._print("  Usage: note edit <note-id> [--title T] [--tags t1,t2] [--status s] [--sprint S1] [--gh owner/repo#123] [--body B]")
            self._last_exit_code = 1
            return

        note_id = parts[0]
        flags = parts[1] if len(parts) > 1 else ""

        kwargs: dict[str, Any] = {}
        if "--title" in flags:
            idx = flags.index("--title") + 7
            kwargs["title"] = flags[idx:].strip()
        if "--tags" in flags:
            idx = flags.index("--tags") + 6
            tag_str = flags[idx:].split("--")[0].strip() if "--" in flags[idx:] else flags[idx:].strip()
            kwargs["tags"] = [t.strip() for t in tag_str.split(",") if t.strip()]
        if "--status" in flags:
            idx = flags.index("--status") + 8
            kwargs["status"] = flags[idx:].split("--")[0].strip() if "--" in flags[idx:] else flags[idx:].strip()
        if "--sprint" in flags:
            idx = flags.index("--sprint") + 8
            kwargs["sprint"] = flags[idx:].split("--")[0].strip() if "--" in flags[idx:] else flags[idx:].strip()
        if "--gh" in flags:
            idx = flags.index("--gh") + 4
            kwargs["gh"] = flags[idx:].split("--")[0].strip() if "--" in flags[idx:] else flags[idx:].strip()
        if "--body" in flags:
            idx = flags.index("--body") + 6
            kwargs["body"] = flags[idx:].strip()

        if not kwargs:
            self._print("  No changes specified. Use --title, --tags, --status, or --body.")
            self._last_exit_code = 1
            return

        updated = store.update(note_id, **kwargs)
        if updated is None:
            self._print(f"  Note not found: {note_id}")
            self._last_exit_code = 1
            return

        self._print(f"  Updated: {updated.short_id}  {updated.title}")
        self._last_exit_code = 0

    def _note_delete(self, store, rest: str) -> None:
        """Delete a note. Usage: note delete <id>"""
        if not rest.strip():
            self._print("  Usage: note delete <note-id>")
            self._last_exit_code = 1
            return

        if store.delete(rest.strip()):
            self._print(f"  Deleted: {rest.strip()}")
            self._last_exit_code = 0
        else:
            self._print(f"  Note not found: {rest.strip()}")
            self._last_exit_code = 1

    def _note_search(self, store, rest: str) -> None:
        """Search notes. Usage: note search <query>"""
        if not rest.strip():
            self._print("  Usage: note search <query>")
            self._last_exit_code = 1
            return

        results = store.search(rest.strip())
        if not results:
            self._print(f"  No notes matching '{rest.strip()}'")
            return

        for n in results:
            tags_str = f"  [{', '.join(n.tags)}]" if n.tags else ""
            self._print(f"    {n.short_id}  {n.title}{tags_str}")
        self._print(f"\n  {len(results)} result(s)")
        self._last_exit_code = 0

    def _note_today(self, store) -> None:
        """Show today's notes."""
        notes = store.today()
        if not notes:
            self._print("  No notes today.")
            return

        for n in notes:
            tags_str = f"  [{', '.join(n.tags)}]" if n.tags else ""
            status_icon = {"open": "○", "wip": "◐", "done": "●", "blocked": "✕"}.get(n.status, "?")
            self._print(f"    {status_icon} {n.short_id}  {n.title}{tags_str}")
        self._print(f"\n  {len(notes)} note(s) today")
        self._last_exit_code = 0

    def _note_export(self, store, rest: str) -> None:
        """Export all notes. Usage: note export [file.md]"""
        output_path = rest.strip() if rest.strip() else None
        content = store.export_all(output_path=output_path)
        if output_path:
            self._print(f"  Exported {store.count()} notes to {output_path}")
        else:
            self._print(content)
        self._last_exit_code = 0

    def _note_tags(self, store) -> None:
        """List all tags with counts."""
        tag_counts: dict[str, int] = {}
        for note in store.list_notes(limit=9999):
            for tag in note.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        if not tag_counts:
            self._print("  No tags found.")
            return

        for tag, count in sorted(tag_counts.items()):
            self._print(f"    {tag:20s}  {count} note(s)")
        self._last_exit_code = 0

    def _note_status_summary(self, store) -> None:
        """Show notes grouped by status."""
        status_counts: dict[str, int] = {}
        for note in store.list_notes(limit=9999):
            status_counts[note.status] = status_counts.get(note.status, 0) + 1

        if not status_counts:
            self._print("  No notes.")
            return

        icons = {"open": "○", "wip": "◐", "done": "●", "blocked": "✕"}
        for status in ["open", "wip", "done", "blocked"]:
            count = status_counts.get(status, 0)
            if count:
                icon = icons.get(status, "?")
                self._print(f"    {icon} {status:10s}  {count}")
        self._last_exit_code = 0

    def _note_sprint(self, store, rest: str) -> None:
        """Sprint operations. Usage: note sprint <name> [list|report]"""
        parts = rest.split(None, 1)
        sprint_name = parts[0].strip() if parts else ""
        action = parts[1].strip().lower() if len(parts) > 1 else "list"

        if not sprint_name:
            sprints = store.sprints()
            if not sprints:
                self._print("  No sprints found.")
                return
            self._print(f"  Sprints: {', '.join(sprints)}")
            return

        notes = store.list_notes(sprint=sprint_name, limit=9999)
        if not notes:
            self._print(f"  No notes for sprint '{sprint_name}'.")
            return

        if action == "report":
            report = store.sprint_report(sprint_name)
            for line in report.split("\n"):
                self._print(f"  {line}")
        else:
            by_status: dict[str, list] = {}
            for n in notes:
                by_status.setdefault(n.status, []).append(n)
            for status in ["open", "wip", "done", "blocked"]:
                items = by_status.get(status, [])
                if not items:
                    continue
                icons = {"open": "○", "wip": "◐", "done": "●", "blocked": "✕"}
                icon = icons.get(status, "?")
                self._print(f"\n  {icon} {status.upper()} ({len(items)})")
                for n in items:
                    gh_tag = f"  #{n.gh}" if n.gh else ""
                    self._print(f"    {n.short_id}  {n.title}{gh_tag}")
            self._print(f"\n  {len(notes)} note(s) in sprint '{sprint_name}'")
        self._last_exit_code = 0

    def _note_timeline(self, store, rest: str) -> None:
        """Timeline view. Usage: note timeline [--days N] [--tag T] [--status S]"""
        days = 7
        tag: str | None = None
        status: str | None = None
        args = rest.split()
        i = 0
        while i < len(args):
            if args[i] == "--days" and i + 1 < len(args):
                try:
                    days = int(args[i + 1])
                except ValueError:
                    pass
                i += 2
            elif args[i] == "--tag" and i + 1 < len(args):
                tag = args[i + 1]
                i += 2
            elif args[i] == "--status" and i + 1 < len(args):
                status = args[i + 1]
                i += 2
            else:
                i += 1
        groups = store.timeline(days=days, tag=tag, status=status)
        if not groups:
            self._print("  No notes in the specified range.")
            return
        total = 0
        icons = {"open": "○", "wip": "◐", "done": "●", "blocked": "✕"}
        for date_str, day_notes in groups:
            self._print(f"\n  ▬ {date_str} ▬")
            for n in day_notes:
                icon = icons.get(n.status, "?")
                tags_s = f"  [{', '.join(n.tags)}]" if n.tags else ""
                sprint_s = f"  [{n.sprint}]" if n.sprint else ""
                self._print(f"    {icon} {n.short_id}  {n.title}{tags_s}{sprint_s}")
            total += len(day_notes)
        self._print(f"\n  {total} note(s) across {len(groups)} day(s)")
        self._last_exit_code = 0

    # ── Command registry ────────────────────────────────────────────
    # AI-domain commands + shell essentials only.
    # cd and pwd are built-in because they must reflect the shell's CWD
    # (subprocess can't change the parent's directory).
    # Other Unix standard commands delegate to system binaries.

    COMMANDS: dict[str, Callable] = {
        "help": _cmd_help,
        "exit": _cmd_exit,
        "cd": _cmd_cd,
        "pwd": _cmd_pwd,
        "echo": _cmd_echo,
        "ls": _cmd_ls,
        "cat": _cmd_cat,
        "mkdir": _cmd_mkdir,
        "rm": _cmd_rm,
        "touch": _cmd_touch,
        "cp": _cmd_cp,
        "mv": _cmd_mv,
        "head": _cmd_head,
        "tail": _cmd_tail,
        "wc": _cmd_wc,
        "grep": _cmd_grep,
        "sort": _cmd_sort,
        "uniq": _cmd_uniq,
        "find": _cmd_find,
        "tee": _cmd_tee,
        "xargs": _cmd_xargs,
        "time": _cmd_time,
        "chmod": _cmd_chmod,
        "du": _cmd_du,
        "diff": _cmd_diff,
        "stat": _cmd_stat,
        "cut": _cmd_cut,
        "tr": _cmd_tr,
        "seq": _cmd_seq,
        "nl": _cmd_nl,
        "fold": _cmd_fold,
        "tac": _cmd_tac,
        "env": _cmd_env,
        "printenv": _cmd_env,
        "yes": _cmd_yes,
        "realpath": _cmd_realpath,
        "dirname": _cmd_dirname,
        "basename": _cmd_basename,
        "nproc": _cmd_nproc,
        "hostname": _cmd_hostname,
        "uname": _cmd_uname,
        "shuf": _cmd_shuf,
        "rev": _cmd_rev,
        "paste": _cmd_paste,
        "comm": _cmd_comm,
        "test": _cmd_test,
        "[": _cmd_test,
        "printf": _cmd_printf,
        "expand": _cmd_expand,
        "unexpand": _cmd_unexpand,
        "id": _cmd_id,
        "logname": _cmd_logname,
        "mktemp": _cmd_mktemp,
        "who": _cmd_who,
        "od": _cmd_od,
        "join": _cmd_join,
        "which": _cmd_which,
        "type": _cmd_type,
        "history": _cmd_history,
        "fc": _cmd_fc,
        "alias": _cmd_alias,
        "unalias": _cmd_unalias,
        "export": _cmd_export,
        "set": _cmd_set,
        "source": _cmd_source,
        ".": _cmd_source,
        "py": _cmd_py,
        "procs": _cmd_procs,
        "ps": _cmd_ps,
        "kill": _cmd_kill,
        "bg": _cmd_bg,
        "jobs": _cmd_bg,
        "fg": _cmd_fg,
        "watch": _cmd_watch,
        "load": _cmd_load,
        "uptime": _cmd_uptime,
        "status": _cmd_status,
        "events": _cmd_events,
        "metrics": _cmd_metrics,
        "train": _cmd_train,
        "gen": _cmd_gen,
        "chat": _cmd_chat,
        "ai": _cmd_ai,
        "agents": _cmd_agents,
        "tutorial": _cmd_tutorial,
        "read": _cmd_read,
        "render": _cmd_render,
        "protect": _cmd_protect,
        "unprotect": _cmd_unprotect,
        "boot": _cmd_boot,
        "shutdown": _cmd_shutdown,
        "svc": _cmd_svc,
        "devices": _cmd_lsdev,
        "lsdev": _cmd_lsdev,
        "asm": _cmd_asm,
        "vmrun": _cmd_vmrun,
        "vmperms": _cmd_vmperms,
        "permit": _cmd_permit,
        "deny": _cmd_deny,
        "permissions": _cmd_permissions,
        "confirm": _cmd_confirm,
        "note": _cmd_note,
        "api": _cmd_api,
        "logs": _cmd_logs,
        "console": _cmd_logs,
        "tui": _cmd_tui,
        "clear": _cmd_clear,
        "sleep": _cmd_sleep,
        "date": _cmd_date,
        "cal": _cmd_cal,
        "ln": _cmd_ln,
    }

    # ── Main loop ───────────────────────────────────────────────────

    def _dispatch(self, line: str) -> None:
        """Execute one input line with full pipeline/background/redirect semantics.

        Shared by the line-mode run loop and the curses TUI so both dispatch
        identically (history, state, audit, pipelines, background jobs).

        Args:
            line: the raw input line to execute.

        Side effects:
            - appends to history, updates state + audit
            - runs the command, writes output via self.console / self._print
            - sets self._last_exit_code / self._aborted
        """
        import time as _time

        self._cmd_count += 1
        self._history.append(line)
        self.state.add_history(line)
        self.state.save()

        self._aborted = False

        commands, is_bg, should_time = self._parse_pipeline(line)

        try:
            if is_bg:
                if len(commands) > 1:
                    self._execute_background_tuples(commands)
                else:
                    self._execute_background(commands[0][0])
                self._audit.command(line, commands[0][0].split()[0] if commands[0][0] else "", line, 0, is_background=True, is_pipeline=len(commands) > 1)
                return
            if len(commands) > 1:
                self._execute_pipeline(commands, should_time=should_time)
                self._audit.command(line, "pipeline", line, self._last_exit_code, is_pipeline=True)
                return

            raw_cmd, op = commands[0]
            expanded = self._expand_alias(raw_cmd)
            parts = expanded.split(maxsplit=1)
            cmd = parts[0].lower()
            args_str = parts[1] if len(parts) > 1 else ""
            handler = self.COMMANDS.get(cmd)
            ext_mod = self._ext_cmds.get(cmd) if handler is None else None

            if handler or ext_mod:
                if not self._check_permission(cmd, args_str, interactive=True):
                    self._last_exit_code = 126
                    self._audit.command(line, cmd, args_str, 126, elapsed_ms=0, expanded=expanded)
                    return
                t0 = _time.time() if should_time else None
                try:
                    if ext_mod:
                        from .console import Console as _Console
                        c = _Console(self.io, has_readline=_HAS_READLINE)
                        self._last_exit_code = ext_mod.run(
                            [cmd] + (args_str.split() if args_str else []),
                            c, self.cmds, self._env,
                        )
                    else:
                        handler(self, args_str)
                        self._last_exit_code = 0
                except SystemExit as e:
                    self._last_exit_code = e.code if isinstance(e.code, int) else 1
                except Exception as e:
                    self._print(f"  {_C_RED}Error:{_C_RESET} {e}")
                    self._last_exit_code = 1
                    self._audit.error(line, repr(e))
                elapsed_ms = (_time.time() - t0) * 1000 if t0 else None
                self._audit.command(line, cmd, args_str, self._last_exit_code, elapsed_ms=elapsed_ms, expanded=expanded)
                if should_time and elapsed_ms is not None:
                    self._print(f"{_C_DIM}  [{elapsed_ms/1000:.2f}s]{_C_RESET}")
            else:
                suggestion = self._suggest_command(cmd)
                msg = f"  {_C_RED}Unknown command:{_C_RESET} {cmd}. Type `help`."
                if suggestion:
                    msg += f" Did you mean `{_C_YELLOW}{suggestion}{_C_RESET}`?"
                self._print(msg)
                self._last_exit_code = 127
                self._audit.unknown(cmd)
        except KeyboardInterrupt:
            self._print(f"  {_C_DIM}Aborted{_C_RESET}")
            self._aborted = True
            self._last_exit_code = 0
        except Exception as e:
            self._print(f"  {_C_RED}Error:{_C_RESET} {e}")
            self._audit.error(line, repr(e))

    def run(self) -> None:
        import signal as _signal
        import logging as _logging

        # Suppress kernel logs from stderr during boot (captured by LogBufferHandler)
        _kernel_logger = _logging.getLogger("slo.kernel")
        _prev_propagate = _kernel_logger.propagate
        _kernel_logger.propagate = False

        # Auto-start API before boot
        api = self.os.api
        if not api.is_running:
            with self.console.spinner("Starting API server") as s:
                result = api.start()
                if result.get("ok"):
                    s.ok("API server ready")
                else:
                    s.fail(f"API start failed: {result.get('error', 'unknown')}")

        boot_log, api_status = self.os.boot()
        _kernel_logger.propagate = _prev_propagate

        self._running = True
        self._status("ok", f"System ready  ({api_status.get('model_id') or 'no model'})" if api_status.get("available") else "API not connected")

        # Split-panel TUI is opt-in: MAN_TUI=1, `sloughgpt shell --tui`, or
        # the `tui` command. Line mode is the default.
        if self._use_tui:
            try:
                from .tui_repl import TuiRepl
                TuiRepl(self, self._log_buffer).run()
            except Exception as e:
                self._print(f"  {_C_RED}TUI unavailable, falling back to line mode:{_C_RESET} {e}")
            self._running = False
            self._audit.shutdown()
            self.state.save()
            self.os.shutdown()
            return

        self._print_header()
        self._audit.startup()

        def _graceful_shutdown(signum, frame):
            self._print(f"\n  {_C_DIM}Signal {signum} received — shutting down gracefully{_C_RESET}")
            self._running = False

        for _sig in (_signal.SIGTERM, _signal.SIGHUP):
            try:
                _signal.signal(_sig, _graceful_shutdown)
            except (OSError, ValueError):
                pass  # signal not available on this platform
        if self.state.first_run:
            self._show_welcome()
        while self._running:
            try:
                prompt = self._render_prompt()
                line = self.io.read(f" {prompt} ")
            except EOFError:
                self._print()
                break
            except KeyboardInterrupt:
                self._print("^C")
                self._last_exit_code = 0
                continue

            # Multiline continuation with trailing backslash
            while line.endswith("\\") and not line.endswith("\\\\"):
                line = line.rstrip("\\").rstrip()
                try:
                    continuation = self.io.read("  > ")
                    line = f"{line} {continuation}"
                except (EOFError, KeyboardInterrupt):
                    break

            if not line:
                continue

            self._dispatch(line)

        self._audit.shutdown()
        self.state.save()
        self.os.shutdown()
