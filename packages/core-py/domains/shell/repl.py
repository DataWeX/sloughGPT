"""
ShellREPL — interactive shell with pipelines, backgrounds, readline,
state persistence, and LLM-powered natural language interpretation.

Features:
  - 40+ built-in commands delegating to real backend endpoints
  - Pipeline chaining (|) — passes captured output as next command's args
  - Background execution (&) — spawns in a thread
  - Command chaining (&&, ||, ;) with exit code tracking ($?)
  - Unix filesystem commands: ls, cd, pwd, mkdir, rm, cp, mv, cat, touch
  - Scripting commands: true, false, test/[, which, type, env
  - readline tab completion for command names
  - Persistent history and aliases (~/.config/sloughgpt/shell_state.json)
  - LLM-powered natural language interpretation (ai <query>)
  - Alias management (alias / unalias)
  - Pipe filters: grep, head, tail, wc
"""

from __future__ import annotations

import io
import os
import re
import sys
import json
import stat
import shutil
import signal
import logging
import fnmatch
import glob
import threading
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .kernel import DaitRuntime
from .commands import ShellCommands
from .state import ShellState

_EM = "\u2014"  # em dash
logger = logging.getLogger("man.shell.repl")

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


# ── Output capture context manager ───────────────────────────────────


class _CaptureOutput:
    """Captures all print() output within a with-block."""

    def __init__(self):
        self._buf = io.StringIO()
        self._old: io.TextIOWrapper | None = None

    def __enter__(self) -> _CaptureOutput:
        self._old = sys.stdout
        sys.stdout = self._buf
        return self

    def __exit__(self, *exc):
        sys.stdout = self._old

    def getvalue(self) -> str:
        return self._buf.getvalue()


# ── REPL ─────────────────────────────────────────────────────────────


class ShellREPL:
    """Interactive REPL with built-in commands and AI-assisted mode."""

    def __init__(self, os: DaitRuntime, cmds: ShellCommands | None = None):
        self.os = os
        self.cmds = cmds or ShellCommands()
        self.state = ShellState()
        self._history: list[str] = self.state.history[:]
        self._running = False
        self._bg_threads: dict[int, threading.Thread] = {}
        self._next_bg_id = 1
        self._piped_input: str = ""
        self._aborted = False
        self._env: dict[str, str] = {
            "PS1": "\u03bb",
            "SHELL": "sloughgpt",
            "HOME": str(Path.home()),
        }
        self._env.update(self.state.env)
        self._update_color_state()

        # Structured logger — inherit from domains.logging
        from domains.logging import ShellLogger, LogLevel
        self.log = ShellLogger("man.shell.repl", level=LogLevel.DEBUG)

        self._aliases: dict[str, str] = dict(self.state.aliases)
        self._aliases.update({
            "q": "exit", "quit": "exit", "h": "help",
            "?": "help", "cls": "clear", "ps": "procs",
            "jobs": "bg",
        })

        self._last_exit_code = 0
        self._cmd_count = 0
        self._dir_stack: list[str] = []

        if _HAS_READLINE:
            self._setup_readline()

        self._load_rc()

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
                    logger.warning("rc line %d: %s", line_no, e)

    def _render_prompt(self) -> str:
        """Expand PS1 escapes: \\h=host, \\w=cwd, \\t=time, \\u=user, \\s=shell, \\#=cmd count."""
        s = self._env.get("PS1", "\u03bb")
        s = s.replace("\\h", os.uname().nodename.split(".")[0])
        s = s.replace("\\w", os.getcwd().replace(str(Path.home()), "~"))
        s = s.replace("\\t", datetime.now().strftime("%H:%M:%S"))
        s = s.replace("\\u", os.environ.get("USER", "user"))
        s = s.replace("\\s", "sloughgpt")
        s = s.replace("\\#", str(self._cmd_count + 1))
        s = s.replace("\\n", "\n")
        if self._last_exit_code != 0:
            s = f"{_C_RED}[{self._last_exit_code}]{_C_RESET} {s}"
        return s

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
            try:
                readline.read_history_file(str(histfile))
            except FileNotFoundError:
                pass
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
            candidates = list(self.COMMANDS.keys()) + list(self._aliases.keys())
        else:
            cmd = parts[0].lower()
            candidates = self._complete_args_for(cmd)

        matches = [c for c in sorted(set(candidates)) if c.startswith(text)]
        try:
            return matches[state]
        except IndexError:
            return None

    def _complete_args_for(self, cmd: str) -> list[str]:
        """Return dynamic completion candidates for a given command."""
        try:
            if cmd in ("load", "unload", "gen"):
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
                ft = self.cmds.finetuned_models()
                return [m.get("model_name", "") for m in ft]
            if cmd == "train":
                # Subcommand completion: status, follow, stop, distill, hf, auto, load, del
                return ["status", "follow", "stop", "distill", "hf", "auto", "load", "del"]
        except Exception:
            pass
        # Fallback: file/directory path completion
        return self._complete_path("")

    def _complete_path(self, prefix: str) -> list[str]:
        """Return matching file/directory paths for tab completion."""
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
        print(*args, **kwargs)

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
        terminal = shutil.get_terminal_size().columns
        sep = f"{_C_DIM}{'━' * terminal}{_C_RESET}"
        lines = self.os.status_summary.split("\n")
        header = lines[0] if lines else ""
        print(sep)
        print(f"{_C_CYAN}{_C_BOLD}  Dait{_C_RESET}".center(terminal))
        print(f"{_C_DIM}  {header}{_C_RESET}".center(terminal))
        print(sep)
        print(f"  Type {_C_YELLOW}`help`{_C_RESET} for commands, {_C_YELLOW}`exit`{_C_RESET} to quit, {_C_YELLOW}`ai <query>`{_C_RESET} for AI mode")
        print()

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
            pipe_parts = [p.strip() for p in seg.split("|")]
            for pp in pipe_parts[:-1]:
                commands.append((pp, '|'))
            commands.append((pipe_parts[-1], op))

        return commands, bg, should_time

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
        if handler is None:
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

        with _CaptureOutput() as cap:
            try:
                handler(self, args)
                self._last_exit_code = 0
            except SystemExit as e:
                self._last_exit_code = e.code if isinstance(e.code, int) else 1
            except Exception as e:
                self._print(f"  Error: {e}")
                self._last_exit_code = 1

        for k in inline_env:
            if old_env[k] is None:
                self._env.pop(k, None)
            else:
                self._env[k] = old_env[k]

        self._piped_input = ""
        output = cap.getvalue()

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
        """Suggest a close command match via difflib."""
        import difflib
        all_cmds = list(self.COMMANDS.keys()) + list(self._aliases.keys())
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
                    print(f"\n[bg-{bg_id}] {out}", end="")
            except Exception as e:
                print(f"\n[bg-{bg_id}] Error: {e}")

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
                print(f"\n[bg-{bg_id}] Error: {e}")

        t = threading.Thread(target=_run, daemon=True, name=f"shell-bg-{bg_id}")
        t.start()
        self._bg_threads[bg_id] = t
        cmds_str = " | ".join(c[0] for c in commands)
        self._print(f"  [bg-{bg_id}] {cmds_str}")

    # ── Pipe-filter commands ────────────────────────────────────────

    def _cmd_grep(self, args: str = "") -> None:
        pattern = args if args else ""
        data = self._piped_input if self._piped_input else ""
        if not pattern and not data:
            self._print("  Usage: grep <pattern>   (filters piped input)")
            return
        if not data and pattern:
            data = pattern
            pattern = ""
        if not pattern:
            self._print("  No pattern specified. Usage: grep <pattern>")
            return
        lines = data.split("\n")
        try:
            matched = [l for l in lines if l and re.search(pattern, l)]
            for m in matched:
                self._print(m)
        except re.error as e:
            self._print(f"  Invalid regex: {e}")

    def _cmd_head(self, args: str = "") -> None:
        n = 10
        data = self._piped_input if self._piped_input else ""
        a = args.strip()
        if a.lstrip("-").isdigit():
            n = abs(int(a))
        elif a.startswith("-n") and len(a) > 2:
            n = int(a[2:].strip())
        elif a and not data:
            data = a
        lines = [l for l in data.split("\n") if l] if data else []
        for l in lines[:n]:
            self._print(l)

    def _cmd_tail(self, args: str = "") -> None:
        n = 10
        data = self._piped_input if self._piped_input else ""
        a = args.strip()
        if a.lstrip("-").isdigit():
            n = abs(int(a))
        elif a.startswith("-n") and len(a) > 2:
            n = int(a[2:].strip())
        elif a and not data:
            data = a
        lines = [l for l in data.split("\n") if l] if data else []
        for l in lines[-n:]:
            self._print(l)

    def _cmd_wc(self, args: str = "") -> None:
        text = self._piped_input if self._piped_input else args
        if not text:
            self._print("  0  0  0")
            return
        line_count = len(text.split("\n"))
        word_count = len(text.split())
        char_count = len(text)
        self._print(f"  {line_count:>4} {word_count:>4} {char_count:>4}")

    def _cmd_echo(self, args: str = "") -> None:
        """Print arguments (like Unix echo). Supports -n (no newline) and -e (escape codes)."""
        if not args:
            self._print()
            return
        parts = args.split()
        no_newline = False
        interpret_escapes = False
        idx = 0
        while idx < len(parts):
            if parts[idx] == "-n":
                no_newline = True
                idx += 1
            elif parts[idx] == "-e":
                interpret_escapes = True
                idx += 1
            elif parts[idx] == "-E":
                interpret_escapes = False
                idx += 1
            else:
                break
        text = " ".join(parts[idx:]) if idx < len(parts) else ""
        if interpret_escapes:
            text = text.encode().decode("unicode_escape")
        self._print(text, end="" if no_newline else "\n")

    def _cmd_tee(self, args: str = "") -> None:
        """Write piped input to file AND pass through to stdout."""
        data = self._piped_input if self._piped_input else ""
        path = args.strip() if args else ""
        if not path or not data:
            self._print("  Usage: <cmd> | tee <file>")
            return
        try:
            with open(os.path.expanduser(path), "w") as f:
                f.write(data)
            self._print(data, end="")
        except OSError as e:
            self._print(f"  Error: {e}")

    def _cmd_sort(self, args: str = "") -> None:
        """Sort lines of piped input. Flags: -r (reverse), -u (unique), -n (numeric)."""
        data = self._piped_input if self._piped_input else args
        if not data:
            return
        flags = set(args.strip().split())
        reverse = bool(flags & {"-r", "--reverse"})
        unique = bool(flags & {"-u", "--unique"})
        numeric = bool(flags & {"-n", "--numeric"})
        lines = [l for l in data.split("\n") if l]
        if numeric:
            def _key(s):
                m = re.search(r"[-+]?\d+\.?\d*", s)
                return float(m.group()) if m else 0.0
            lines = sorted(lines, key=_key, reverse=reverse)
        else:
            lines = sorted(lines, reverse=reverse)
        if unique:
            seen = set()
            deduped = []
            for l in lines:
                if l not in seen:
                    seen.add(l)
                    deduped.append(l)
            lines = deduped
        for l in lines:
            self._print(l)

    def _cmd_uniq(self, args: str = "") -> None:
        """Deduplicate consecutive lines of piped input."""
        data = self._piped_input if self._piped_input else args
        if not data:
            return
        lines = data.split("\n")
        prev = None
        for l in lines:
            if l != prev:
                self._print(l)
            prev = l

    def _cmd_less(self, args: str = "") -> None:
        """Pager — scroll through piped output (Enter=next page, q=quit)."""
        data = self._piped_input if self._piped_input else args
        if not data:
            self._print("  Usage: <cmd> | less   (pager: Enter=next page, q=quit)")
            return
        lines = data.rstrip("\n").split("\n")
        rows, _cols = shutil.get_terminal_size()
        page_size = max(rows - 3, 1)
        pos = 0
        while pos < len(lines):
            for l in lines[pos:pos + page_size]:
                self._print(l)
            pos += page_size
            if pos < len(lines):
                try:
                    resp = input(f"{_C_DIM}-- more ({pos}/{len(lines)} lines) --{_C_RESET} ")
                    if resp.strip().lower() == "q":
                        break
                except (EOFError, KeyboardInterrupt):
                    break

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
        """Evaluate a Python expression and print the result."""
        if not args:
            self._print("  Usage: py <expression>")
            self._print("  Example: py 2 + 2")
            self._print("  Example: py [i*i for i in range(5)]")
            self._print("  Example: py __import__('json').dumps({'a': 1})")
            return
        try:
            result = eval(args, {"__builtins__": __builtins__})
            self._print(repr(result))
        except Exception as e:
            self._print(f"  Error: {e}")

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

    def _cmd_sleep(self, args: str = "") -> None:
        """Pause for N seconds."""
        if not args:
            self._print("  Usage: sleep <seconds>")
            return
        try:
            n = float(args.strip())
            import time as _time
            self._print(f"  Sleeping for {n}s...")
            _time.sleep(n)
            self._print("  Done")
        except ValueError:
            self._print(f"  Invalid number: {args}")

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

    def _cmd_pushd(self, args: str = "") -> None:
        """Push current directory onto stack and cd to <dir>."""
        if not args:
            self._print("  Usage: pushd <dir>")
            return
        path = os.path.expanduser(args.strip())
        if not os.path.isdir(path):
            self._print(f"  Not a directory: {path}")
            return
        self._dir_stack.append(os.getcwd())
        os.chdir(path)
        cwd = os.getcwd().replace(str(Path.home()), "~")
        self._print(f"  {cwd}")

    def _cmd_popd(self, args: str = "") -> None:
        """Pop directory stack and cd back."""
        if not self._dir_stack:
            self._print("  Directory stack is empty")
            return
        path = self._dir_stack.pop()
        os.chdir(path)
        cwd = os.getcwd().replace(str(Path.home()), "~")
        self._print(f"  {cwd}")

    def _cmd_dirs(self, args: str = "") -> None:
        """Show directory stack."""
        if not self._dir_stack:
            self._print("  (empty)")
            return
        cwd = os.getcwd().replace(str(Path.home()), "~")
        self._print(f"  {cwd}  (current)")
        for i, d in enumerate(reversed(self._dir_stack), 1):
            display = d.replace(str(Path.home()), "~")
            self._print(f"  {i}: {display}")

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

    def _cmd_help(self, args: str = "") -> None:
        if args:
            if args == "brief":
                self._print("""
  Most-used commands (help <cmd> for details, help for full list):

  ls / cd / pwd / cat   File system navigation
  models / load / unload  Model management
  souls / switch        Soul personality management
  gen / tokenizer       Inference
  health / status / uname  System info
  boot / shutdown       Shell lifecycle
  svc                   Service management
  devices / lsdev       AI device nodes (/dev/llm, /dev/embedding, /dev/knowledge)
  asm / wm              Virtual machine / window manager
  remember / recall     Knowledge base
  procs / kill / bg / fg  Process management
  py / ai / echo        Shell utilities
  history / fc          Command history
  help <cmd>            Help for a specific command
  exit / q / quit       Exit shell

  Pipe features: |  &  >  >>  $(...)  $?  $VAR
""")
                return
            cmd_help = {
                "help": "  help [cmd]  — Show this help or help for a specific command",
                "exit": "  exit | q | quit  — Exit the shell",
                "clear": "  clear | cls  — Clear the screen",
                "history": "  history [n]  — Show command history (last n entries, default 20)",
                "fc": "  fc [-l] [n]  — List history, or re-run command by number (fc 42)",
                "alias": "  alias [name=cmd]  — List or set aliases",
                "unalias": "  unalias <name>  — Remove an alias",
                "export": "  export  — Show shell state (history count, aliases, env vars)",
                "set": '  set [name=value]  — Set/show env vars. $VAR, ${VAR}, and NAME=VALUE cmd supported',
                "sleep": "  sleep <sec>  — Pause for N seconds",
                "source": "  source <file> | . <file>  — Execute commands from a file",
                "sleep": "  sleep <sec>  — Pause for N seconds",
                "ls": "  ls [-l] [-a] [path]  — List directory contents",
                "cd": "  cd [dir]  — Change directory (no arg = $HOME, - = previous)",
                "pwd": "  pwd  — Print working directory",
                "mkdir": "  mkdir [-p] <dir>  — Create directory",
                "rm": "  rm [-r] [-f] <path>  — Remove files/directories",
                "cp": "  cp [-r] <src> <dst>  — Copy files/directories",
                "mv": "  mv <src> <dst>  — Move/rename files",
                "cat": "  cat <file>  — Print file contents",
                "touch": "  touch <file>  — Create/update file timestamps",
                "chmod": "  chmod <mode> <file>  — Change file permissions",
                "find": "  find [path] -name <glob>  — Recursively find files matching a pattern",
                "true": "  true  — Return exit code 0",
                "false": "  false  — Return exit code 1",
                "test": "  test <expr> | [ <expr> ]  — Evaluate expression",
                "which": "  which <command>  — Locate a command",
                "type": "  type <command>  — Describe a command",
                "env": "  env  — Print environment variables",
                "font": "  font [name]  — Show or set terminal font (OSC 50)",
                "watch": "  watch <sec> <cmd>  — Run command repeatedly every N seconds",
                "pushd": "  pushd <dir>  — Push dir onto stack and cd there",
                "popd": "  popd  — Pop dir stack and cd back",
                "dirs": "  dirs  — Show directory stack",
                "export": "  export [NAME=VALUE]  — Set/show env vars (POSIX-style)",
                "tee": "  tee <file>  — Write piped input to file + pass through",
                "sort": "  sort [-r] [-u] [-n]  — Sort piped lines; -r=reverse, -u=unique, -n=numeric",
                "uniq": "  uniq  — Deduplicate consecutive piped lines",
                "less": "  less  — Pager: scroll through piped output page by page",
                "procs": "  procs | ps  — List running training jobs",
                "ps": "  procs | ps  — List running training jobs",
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
                "uname": "  uname [-a]  — Print system information (Dait version, machine)",
                "uptime": "  uptime  — How long Dait has been running",
                "health": "  health  — Quick API health check (colored status output)",
                "status": "  status  — Detailed system status (model, soul, server)",
                "metrics": "  metrics  — Show CPU/memory/disk metrics from server",
                "datasets": "  datasets  — List datasets (tab-completes names)",
                "knowledge": "  knowledge [query]  — List/search knowledge base entries",
                "remember": "  remember <fact>  — Store a fact in the knowledge base",
                "recall": "  recall <query>  — Search the knowledge base",
                "checkpoints": "  checkpoints  — List training checkpoints (tab-completes names)",
                "finetuned": "  finetuned  — List fine-tuned models",
                "train": "  train [dataset] | train status | train follow <id> | train stop <id>  — Training operations",
                "finetuned": "  finetuned  — List fine-tuned model paths (tab-completes names)",
                "gen": "  gen <prompt>  — Generate text via inference",
                "tokenizer": "  tokenizer  — Show tokenizer vocabulary stats",
                "py": '  py <expr>  — Evaluate a Python expression. E.g. py 2 + 2, py [i*2 for i in range(5)]',
                "grep": "  grep <pattern>  — Filter piped lines by regex",
                "head": "  head [n]  — Show first n lines of piped input (default 10)",
                "tail": "  tail [n]  — Show last n lines of piped input (default 10)",
                "wc": "  wc  — Count lines/words/characters of piped input",
                "echo": "  echo <text>  — Print text",
                "ai": '  ai <query>  — LLM-powered NL interpretation. E.g. ai "show me running jobs"',
                "agents": "  agents <goal>  — Multi-agent orchestration. E.g. agents 'research and write about X'",
                "agents_list": "  agents list  — List available specialized agents",
                "tutorial": "  tutorial  — Interactive walkthrough of shell features",
                "wm": "  wm [split-h|split-v|close|layout]  — Open window manager TUI, or manage panes from the shell",
                "pbcopy": "  pbcopy  — Copy piped input to macOS clipboard",
                "pbpaste": "  pbpaste  — Paste from macOS clipboard",
                "remember": "  remember <fact>  — Store a fact in the knowledge base (also piped input)",
                "recall": "  recall <query>  — Search the knowledge base",
                "boot": "  boot  — Boot the shell (kernel + init services)",
                "shutdown": "  shutdown  — Halt all services and kernel",
                "svc": "  svc [list|start|stop|restart|status] [name]  — Manage init services",
                "devices": "  devices | lsdev  — List AI device nodes (/dev/*)",
                "lsdev": "  devices | lsdev  — List AI device nodes (/dev/*)",
                "asm": '  asm [file.asm] | asm --test | asm --list  — Assemble and run VM programs',
            }
            if args in cmd_help:
                self._print(cmd_help[args])
            elif self.COMMANDS.get(args):
                self._print(f"  {args}  — (built-in command)")
            elif args == "brief":
                self._print("""
Most common commands (help [cmd] for details, help for full list):
  help [cmd]           Show help
  exit / q / quit      Exit the shell
  clear / cls          Clear screen
  history [n]          Show command history
  ls / cd / pwd        Navigate filesystem
  cat <file>           Print file
  mkdir / rm / cp / mv  File operations
  find -name <glob>    Find files
  gen <prompt>         Generate text
  models               List models
  load <name>          Load model
  souls                List souls
  switch <name>        Switch soul
  whoami               Current soul
  health / status / uname  System info
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
  grep / head / tail / wc  Pipe filters
  pbcopy / pbpaste     Clipboard
""")
            else:
                self._print(f"  Unknown command: {args}")
            return
        self._print("""
Built-in commands:
  help [cmd]             Show this help or help for a specific command
  exit / q / quit         Exit the shell
  clear / cls             Clear the screen
  history [n]             Show command history
  fc [-l] [n]             List history, or re-run command #n (fc 42)
  alias [name=cmd]        List or set aliases
  unalias <name>          Remove an alias
  export [NAME=VALUE]     Set/show env vars (POSIX-style)
  set [name=value]        Set/show environment variables ($VAR expansion)
  source <file> / .       Execute commands from a file
  py <expr>               Evaluate a Python expression
  sleep <sec>             Pause for N seconds
  watch <sec> <cmd>       Run command repeatedly every N seconds
  pushd <dir>             Push dir onto stack and cd there
  popd                    Pop dir stack and cd back
  dirs                    Show directory stack

Unix filesystem:
  ls [-l] [-a] [path]    List directory contents
  cd [dir]                Change directory
  pwd                     Print working directory
  mkdir [-p] <dir>        Create directory
  rm [-r] [-f] <path>     Remove files/directories
  cp [-r] <src> <dst>     Copy files/directories
  mv <src> <dst>          Move/rename files
  cat <file>              Print file contents
  touch <file>            Create/update file timestamps
  chmod <mode> <file>     Change file permissions
  find [path] -name <glob>  Recursively find files matching a pattern

Clipboard (macOS):
  pbcopy                  Copy piped input to clipboard. E.g. gen hello | pbcopy
  pbpaste                 Paste from clipboard

Knowledge:
  remember <fact>         Store a fact in the knowledge base
  recall <query>          Search the knowledge base

Unix scripting:
  true                    Return exit code 0
  false                   Return exit code 1
  test <expr> / [ <expr> ]  Evaluate expression (-f, -d, -e, =, !=)
  which <cmd>             Locate a command
  type <cmd>              Describe a command
  env                     Print environment variables
  font [name]             Show or set terminal font (OSC 50 / iTerm2)

Process management:
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

Init system:
  boot                    Boot the shell (kernel + services)
  shutdown                Halt all services + kernel
  svc                     Service manager: list, start, stop, restart, status

Devices:
  devices / lsdev         List AI device nodes (/dev/*)
  cat /dev/<name>         Read from an AI device (e.g. cat /dev/llm, cat /dev/random)

Model management:
  models                  List available models (tab-completes names)
  load <name>             Load a model (tab-completes names)
  unload                  Unload the current model

Souls:
  souls                   List available souls (tab-completes names)
  switch <name>           Switch to a soul (tab-completes names)
  whoami                  Show current soul

System:
  health                  Quick health check (colored status)
  status                  Detailed system status
  metrics                 CPU/memory/disk metrics
  uname [-a]              Print system info (Dait version, machine)
  uptime                  How long Dait has been running

Data:
  datasets                List datasets (tab-completes names)
  knowledge               List knowledge base entries

Training:
  checkpoints             List training checkpoints (tab-completes names)
  finetuned               List fine-tuned models (tab-completes names)

Inference:
  gen <prompt>            Generate text
  tokenizer               Show tokenizer stats

Pipe filters:
  grep <pattern>          Filter lines by regex
  head [n]                Show first n lines (default 10)
  tail [n]                Show last n lines (default 10)
  wc                      Count lines/words/chars
  tee <file>              Write piped input to file + pass through
  sort [-r] [-u] [-n]     Sort piped lines; -r=reverse, -u=unique, -n=numeric
  uniq                    Deduplicate consecutive piped lines
  less                    Pager: scroll through output page by page
  echo <text>             Print text

Shell features:
  <cmd> | <cmd>           Pipeline: output of first feeds second
  <cmd> &                 Background: run without blocking
  <cmd> && <cmd>          Chain: run next only if previous succeeded ($?=0)
  <cmd> || <cmd>          Chain: run next only if previous failed ($?!=0)
  <cmd> ; <cmd>           Chain: run next regardless of exit code
  <cmd> > <file>          Redirect output to file (overwrite)
  <cmd> > /dev/llm        Redirect output to AI device (write)
  <cmd> >> <file>         Redirect output to file (append)
  /dev/llm                AI device node: cat /dev/llm, echo hi > /dev/llm
  /dev/null               Discard output: <cmd> > /dev/null
  /dev/random             Random tokens: cat /dev/random
  /dev/embedding          Compute embeddings: echo text > /dev/embedding
  /dev/knowledge          Knowledge base: cat /dev/knowledge, echo fact > /dev/knowledge

Virtual machine:
  asm [file.asm]          Assemble and run a VM program (.text + .data sections)
  asm --test              Run VM self-tests
  time <cmd>              Show command execution time
  $?                      Exit code of last command
  ai <query>              LLM-powered natural language interpretation
  agents <goal>           Multi-agent orchestration (researcher + writer + critic)
  agents list             List available agents
  wm [subcmd]             Open window manager TUI (:split-h, :split-v, :close, :q)
  $(cmd)                  Command substitution: inline output of cmd
  py <expr>               Evaluate Python expression
  $VAR / ${VAR}           Environment variable expansion
  NAME=VALUE cmd          Inline env var (set for single command)
  \\\\h \\\\w \\\\t \\\\u \\\\#    PS1 escapes: host, cwd, time, user, cmd#

Examples:
  ls -la
  cd /tmp && ls
  test -f file.txt && cat file.txt
  mkdir -p build && cd build
  false || echo "failed"
  models | grep gpt
  health &
  gen hello > output.txt
  time load gpt2
  ai show me running training jobs
  set PS1=$  ;  echo $HOME
  echo $?
  alias ll=procs
  source setup.sh
  py 2 + 2
"""[:-1])

    def _cmd_clear(self, args: str = "") -> None:
        os.system("clear" if os.name == "posix" else "cls")

    def _cmd_exit(self, args: str = "") -> None:
        self._running = False
        self.state.save()
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

    def _cmd_procs(self, args: str = "") -> None:
        jobs = self.cmds.ps()
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
        self._print(self._format_table(rows, ["ID", "Status", "Name", "Progress", "Loss"]))

    def _cmd_kill(self, args: str = "") -> None:
        if not args:
            self._print("  Usage: kill <job_id>")
            return
        result = self.cmds.kill(args.strip())
        self._print(self._dump_json(result))

    def _cmd_models(self, args: str = "") -> None:
        models = self.cmds.models()
        if not models:
            self._print("  No models available")
            return
        rows = []
        for m in models:
            name = m.get("model_id", m.get("name", m.get("id", "?")))
            sz = m.get("size_gb", 0) or m.get("size_mb", 0) / 1024
            sz_str = f"{sz:.2f}G" if sz else ""
            loaded = m.get("status") == "loaded" or m.get("loaded")
            rows.append([name, m.get("type", ""), sz_str, "\u2713 loaded" if loaded else ""])
        self._print(self._format_table(rows, ["Model", "Type", "Size", "Status"]))

    def _cmd_load(self, args: str = "") -> None:
        if not args:
            self._print("  Usage: load <model_name>")
            return
        model_name = args.strip()
        self._print(f"  Loading {model_name}...")
        self._print("  (this may take 30-120s on CPU)")
        result = self.cmds.load_model(model_name)
        status = result.get("status", "?")
        if status == "loaded":
            self._print(f"  ✓ {model_name} loaded on {result.get('device', 'cpu')}")
        elif status == "error":
            self._print(f"  ✗ {result.get('error', 'Unknown error')}")
        else:
            self._print(self._dump_json(result))

    def _cmd_unload(self, args: str = "") -> None:
        result = self.cmds.unload_model()
        self._print(self._dump_json(result))

    def _cmd_souls(self, args: str = "") -> None:
        souls = self.cmds.souls()
        if not souls:
            self._print("  No souls available")
            return
        rows = []
        for s in souls:
            name = s.get("name", s.get("id", "?"))
            desc = s.get("description", "")[:50]
            traits = s.get("traits", [])
            trait_str = ", ".join(str(t)[:15] for t in traits[:3])
            rows.append([name, desc, trait_str])
        self._print(self._format_table(rows, ["Soul", "Description", "Traits"]))

    def _cmd_switch(self, args: str = "") -> None:
        if not args:
            self._print("  Usage: switch <soul_name>")
            return
        result = self.cmds.switch_soul(args.strip())
        self._print(self._dump_json(result))

    def _cmd_whoami(self, args: str = "") -> None:
        soul = self.cmds.current_soul()
        self._print(f"  Current soul: {soul.get('name', 'unknown')}")
        if soul.get("description"):
            self._print(f"  Description: {soul['description']}")

    def _cmd_uname(self, args: str = "") -> None:
        """Print system information (like Unix uname)."""
        a = args.strip()
        if not a:
            self._print("Dait")
            return
        sysname = "Dait"
        nodename = os.uname().nodename
        release = "0.1"
        version = "Dait 0.1"
        machine = os.uname().machine
        if a == "-a" or a == "--all":
            self._print(f"{sysname} {nodename} {release} {version} {machine}")
        elif a == "-s" or a == "--kernel-name":
            self._print(sysname)
        elif a == "-n" or a == "--nodename":
            self._print(nodename)
        elif a == "-r" or a == "--kernel-release":
            self._print(release)
        elif a == "-v" or a == "--kernel-version":
            self._print(version)
        elif a == "-m" or a == "--machine":
            self._print(machine)
        else:
            self._print(f"  uname: invalid option -- {a}")
            self._last_exit_code = 1

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

    def _cmd_health(self, args: str = "") -> None:
        h = self.cmds.health()
        status = h.get("status", "unknown")
        colored = f"{_C_GREEN}{status}{_C_RESET}" if status == "healthy" else f"{_C_YELLOW}{status}{_C_RESET}"
        self._print(f"  Status: {colored}")
        self._print(f"  Model:  {h.get('model_type', _EM)}")
        self._print(f"  Soul:   {h.get('soul_name', _EM)}")

    def _cmd_status(self, args: str = "") -> None:
        self._print(self.os.status_summary)
        try:
            detailed = self.cmds.health_detailed()
            if isinstance(detailed, dict) and "registry" in detailed:
                registry = detailed.get("registry", {})
                models = registry.get("models", []) or registry.get("names", [])
                if models:
                    self._print(f"  Registry models: {len(models)}")
        except Exception:
            pass

    def _cmd_metrics(self, args: str = "") -> None:
        metrics = self.cmds.system_metrics()
        if metrics.get("error"):
            self._print(f"  Error: {metrics['error']}")
            return
        for k, v in metrics.items():
            if not k.startswith("_"):
                self._print(f"  {k}: {v}")

    def _cmd_datasets(self, args: str = "") -> None:
        datasets = self.cmds.datasets()
        if not datasets:
            self._print("  No datasets available")
            return
        rows = []
        for d in datasets:
            name = d.get("name", "?")
            samples = d.get("samples", 0)
            sz = d.get("size", 0)
            sz_str = f"{sz / 1048576:.1f}M" if sz else ""
            rows.append([name, str(samples), sz_str])
        self._print(self._format_table(rows, ["Dataset", "Samples", "Size"]))

    def _cmd_knowledge(self, args: str = "") -> None:
        """List/search knowledge base entries."""
        if args:
            results = self.cmds.list_knowledge(args)
            if not results:
                self._print("  No results")
                return
            for r in results[:20]:
                self._print(f"  \u2022 {r.get('content', '')[:120]}")
            return
        stats = self.cmds.knowledge_stats()
        count = stats.get("total_items", 0)
        if count == 0:
            self._print("  Knowledge base is empty")
            self._print("  Use: remember <fact>  to add a fact")
            return
        self._print(f"  Knowledge base: {count} fact(s)")
        topics = stats.get("topics", {})
        if topics:
            self._print(f"  Topics: {', '.join(sorted(topics.keys()))}")

    def _cmd_remember(self, args: str = "") -> None:
        """Store a fact in the knowledge base. Supports piped input."""
        if not args and not self._piped_input:
            self._print("  Usage: remember <fact>")
            self._print("    remember this project uses FastAPI")
            self._print("    cat notes.txt | remember")
            self._last_exit_code = 1
            return
        content = self._piped_input.strip() if self._piped_input else args
        result = self.cmds.add_knowledge(content)
        if isinstance(result, dict) and result.get("status") == "stored":
            topic = result.get("topic", "general")
            preview = content[:80].replace("\n", "\\n")
            self._print(f"  Stored fact [{topic}]: {preview}...")
        else:
            self._print(f"  Error: {result}")

    def _cmd_recall(self, args: str = "") -> None:
        """Search the knowledge base. Shows recent facts if no query."""
        if not args:
            stats = self.cmds.knowledge_stats()
            count = stats.get("total_items", 0)
            if count == 0:
                self._print("  Knowledge base is empty")
                return
            self._print(f"  Knowledge base: {count} fact(s)")
            topics = stats.get("topics", {})
            if topics:
                self._print(f"  Topics: {', '.join(sorted(topics.keys()))}")
            self._print("  Use: recall <query>  to search")
            return
        results = self.cmds.list_knowledge(args)
        if not results:
            self._print("  No matching facts")
            return
        for r in results[:10]:
            topic = r.get("topic", "")
            content = r.get("content", "")
            score = r.get("score", 0)
            self._print(f"  [{topic}] {content[:140]}  (score: {score:.2f})")
        if len(results) > 10:
            self._print(f"  ... and {len(results) - 10} more")

    def _cmd_checkpoints(self, args: str = "") -> None:
        cps = self.cmds.checkpoints()
        if not cps:
            self._print("  No checkpoints")
            return
        rows = []
        for cp in cps:
            rows.append([cp.get("name", ""), f"{cp.get('loss', _EM)}", cp.get("model_type", "")])
        self._print(self._format_table(rows, ["Checkpoint", "Loss", "Type"]))

    def _cmd_finetuned(self, args: str = "") -> None:
        models = self.cmds.finetuned_models()
        if not models:
            self._print("  No fine-tuned models")
            return
        rows = []
        for m in models:
            name = m.get("model_name", "")
            loss = m.get("final_loss", "\u2014")
            ep = m.get("epochs", 0)
            sz_bytes = m.get("size_bytes", 0)
            sz_str = f"{sz_bytes / 1048576:.0f}M"
            rows.append([name, f"{loss}", f"{ep}ep", sz_str])
        self._print(self._format_table(rows, ["Model", "Loss", "Epochs", "Size"]))

    def _cmd_train(self, args: str = "") -> None:
        """Train: train [dataset] | train status | train follow <id> | train stop <id> | train distill <dataset>"""
        parts = args.strip().split()
        sub = parts[0] if parts else ""

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
            self._print(self._format_table(rows, ["ID", "Status", "Model", "Progress"]))
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
            r = self.cmds.train_distill(dataset, teacher=teacher, epochs=epochs)
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
            r = self.cmds.train_hf(model, dataset, epochs=epochs)
            if "error" in r:
                self._print(f"  Error: {r['error']}")
            else:
                job_id = r.get("id", "")
                self._print(f"  Fine-tuning started: {r.get('status', r)}")
                if job_id:
                    self._stream_train_progress(job_id)
            return

        if sub == "auto":
            soul = parts[1] if len(parts) > 1 else ""
            teacher = parts[2] if len(parts) > 2 else "gpt2"
            epochs = int(parts[3]) if len(parts) > 3 else 10
            r = self.cmds.train_auto(soul_name=soul, teacher=teacher, epochs=epochs)
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
        r = self.cmds.train_quick(dataset, name=name)
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

        while True:
            try:
                result = _api_get(f"/training/jobs/{job_id}")
                if not result:
                    self._print(f"  Job {job_id} not found")
                    return

                status = result.get("status", "unknown")
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

    def _cmd_gen(self, args: str = "") -> None:
        if not args:
            self._print("  Usage: gen <prompt>")
            return
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
            self._print("  [session cleared]")
            return
        if not hasattr(self, '_chat_session_id') or not self._chat_session_id:
            import uuid
            self._chat_session_id = str(uuid.uuid4())
            self._chat_history: list[dict[str, str]] = []
            self._print("  [new session]")
        self._chat_history.append({"role": "user", "content": args})
        result = self.cmds.chat(self._chat_history)
        if isinstance(result, dict) and "message" in result:
            text = result["message"]
            text = text.replace("<think>", "").replace("</think>", "")
            self._print(f"\n  {text.strip()}\n")
            self._chat_history.append({"role": "assistant", "content": text})
        elif isinstance(result, dict) and "error" in result:
            self._print(f"  Error: {result['error']}")
        else:
            self._print(self._dump_json(result))

    def _cmd_tokenizer(self, args: str = "") -> None:
        stats = self.cmds.tokenizer_stats()
        if isinstance(stats, dict) and "error" not in stats:
            for k, v in stats.items():
                self._print(f"  {k}: {v}")
        else:
            self._print(f"  {self._dump_json(stats)}")

    # ── LLM-powered NL interpreter ──────────────────────────────────

    def _cmd_wm(self, args: str = "") -> None:
        """Enter the window manager TUI. Subcommands: split-h, split-v, close, layout."""
        wm = get_window_manager(parent_shell=self)
        parts = args.strip().split()
        verb = parts[0].lower() if parts else ""

        def _layout_str() -> str:
            ws = wm._workspace
            return f"  Layout: {ws.layout} | Panes: {len(ws.panes)} | Focus: {ws.focus_idx}"

        if verb == "split-h":
            wm.split_horizontal()
            self._print(_layout_str())
        elif verb == "split-v":
            wm.split_vertical()
            self._print(_layout_str())
        elif verb == "close":
            title = wm.close_pane()
            if title:
                self._print(f"  Closed: {title}")
        elif verb == "layout":
            self._print(_layout_str())
        elif verb == "reset":
            from .window_manager import reset_window_manager as _rw
            _rw()
            self._print("  Window manager state reset.")
        else:
            # Enter the curses TUI
            try:
                wm.run()
            except Exception as e:
                self._print(f"  Window manager error: {e}")
            finally:
                self._print("  Exited window manager.")

    def _cmd_agents(self, args: str = "") -> None:
        """Multi-agent orchestration: agents <goal> or agents list."""
        from domains.agents.multi import get_orchestrator
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
            self._print("  agent add: not yet implemented")
        elif not args:
            self._cmd_help("agents")
        else:
            goal = args.strip()
            self._print(f"  \U0001f916 Orchestrating agents for: {goal}")
            self._print(f"  {_C_DIM}Planning...{_C_RESET}")
            try:
                result = orch.execute(goal)
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

        available_commands = "\n".join(
            f"  {name} - {cmd.__doc__ or ''}"
            for name, cmd in sorted(self.COMMANDS.items())
        )

        prompt = (
            "You are an AI shell assistant. Given the available commands below, "
            "interpret the user's natural language request and respond with ONLY "
            "the exact shell command to run. Do NOT include any explanation, "
            "backticks, or extra text. Just the command.\n\n"
            f"Available commands:\n{available_commands}\n\n"
            f"User request: {args}\n\n"
            "Command:"
        )

        self._print(f"  \u2601\ufe0f Interpreting as LLM query...")
        result = self.cmds.generate(prompt, max_tokens=60)

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
            error = result.get("error", "unknown")
            self._print(f"  AI interpretation failed: {error}")
            self._print("  Falling back to keyword matching...")
            self._interpret_natural(args)

    def _interpret_natural(self, query: str) -> None:
        """Keyword-based NL fallback when LLM is unavailable."""
        q = query.lower()
        if any(w in q for w in ["process", "job", "running", "ps", "procs"]):
            self._cmd_procs()
        elif any(w in q for w in ["model", "models"]):
            self._cmd_models()
        elif any(w in q for w in ["soul", "personality"]):
            self._cmd_whoami()
        elif any(w in q for w in ["health", "status"]):
            self._cmd_health()
        elif any(w in q for w in ["dataset", "data"]):
            self._cmd_datasets()
        elif any(w in q for w in ["knowledge", "fact"]):
            self._cmd_knowledge()
        elif any(w in q for w in ["checkpoint"]):
            self._cmd_checkpoints()
        elif any(w in q for w in ["finetune", "trained"]):
            self._cmd_finetuned()
        elif any(w in q for w in ["metric", "cpu", "memory", "disk"]):
            self._cmd_metrics()
        elif any(w in q for w in ["tokenizer", "vocab"]):
            self._cmd_tokenizer()
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
        w(f"    models | grep gpt    Filter output")
        w(f"    health &             Run in background")
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
        _proceed = lambda: input(f"{_C_DIM}Press Enter to continue, or q to quit...{_C_RESET} ")

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
                "    models | grep gpt",
                "    health | head 3",
                "    echo hello | wc",
                "  Filters: grep, head, tail, wc, sort, uniq, tee, less",
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
                "  Advanced: pipelines, watch, sleep, pushd/popd",
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

    # ── Unix filesystem commands ───────────────────────────────────

    def _cmd_ls(self, args: str = "") -> None:
        """List directory contents (like Unix ls). Supports -l, -a, -h flags."""
        parts = args.strip().split()
        flags = {p for p in parts if p.startswith("-")}
        targets = [p for p in parts if not p.startswith("-") or p == "--"]
        show_all = "-a" in flags or "--all" in flags or "-A" in flags
        long_fmt = "-l" in flags
        human = "-h" in flags
        show_dirs = targets or ["."]

        for dir_idx, dir_path in enumerate(show_dirs):
            path = os.path.expanduser(dir_path)
            vfs = self.os.vfs
            if vfs and (path.startswith("/dev") or path.startswith("/proc")):
                entries = vfs.listdir(path)
                if entries is None:
                    self._print(f"  ls: cannot access '{dir_path}': No such file or directory")
                    self._last_exit_code = 2
                    continue
                if not show_all:
                    entries = [e for e in entries if not e.startswith(".")]
                if long_fmt:
                    for name in entries:
                        st = vfs.stat(os.path.join(path, name))
                        if st:
                            mode = stat.filemode(st.st_mode)
                            sz = st.st_size
                            sz_str = str(sz).rjust(8)
                            self._print(f"  {mode} {st.st_nlink:2d} {sz_str} {name}")
                        else:
                            self._print(f"  ??????????  0 ??? {name}")
                else:
                    self._print("  " + "  ".join(entries))
                continue
            if not os.path.exists(path):
                self._print(f"  ls: cannot access '{dir_path}': No such file or directory")
                self._last_exit_code = 2
                continue
            if len(show_dirs) > 1:
                self._print(f"  {dir_path}:")
            try:
                entries = sorted(os.listdir(path))
            except PermissionError as e:
                self._print(f"  ls: {e}")
                self._last_exit_code = 2
                continue

            if not show_all:
                entries = [e for e in entries if not e.startswith(".")]

            if long_fmt:
                for name in entries:
                    full = os.path.join(path, name)
                    try:
                        st = os.stat(full)
                        mode = stat.filemode(st.st_mode)
                        sz = st.st_size
                        if human:
                            for unit in ['B', 'K', 'M', 'G', 'T']:
                                if sz < 1024:
                                    break
                                sz /= 1024
                            sz_str = f"{sz:.1f}{unit}" if sz >= 10 else f"{sz:.1f}{unit}"
                        else:
                            sz_str = str(st.st_size).rjust(8)
                        self._print(f"  {mode} {st.st_nlink:2d} {sz_str} {name}")
                    except OSError:
                        self._print(f"  ??????????  0 ??? {name}")
            else:
                if human:
                    self._print("  " + "  ".join(entries))
                else:
                    cols = shutil.get_terminal_size().columns
                    col_w = max(len(e) for e in entries) + 2 if entries else 20
                    ncols = max(1, cols // col_w)
                    rows = [entries[i:i + ncols] for i in range(0, len(entries), ncols)]
                    for row in rows:
                        self._print("  " + "".join(e.ljust(col_w) for e in row))

    def _cmd_cd(self, args: str = "") -> None:
        """Change directory (like Unix cd). No args goes to $HOME."""
        target = args.strip() if args else os.environ.get("HOME", "~")
        target = os.path.expanduser(target)
        if target == "-":
            if len(self._dir_stack) > 0:
                target = self._dir_stack.pop()
            else:
                self._print("  cd: no previous directory in stack")
                self._last_exit_code = 1
                return
        try:
            os.chdir(target)
        except FileNotFoundError:
            self._print(f"  cd: no such file or directory: {args}")
            self._last_exit_code = 1
        except PermissionError:
            self._print(f"  cd: permission denied: {args}")
            self._last_exit_code = 1
        except NotADirectoryError:
            self._print(f"  cd: not a directory: {args}")
            self._last_exit_code = 1

    def _cmd_pwd(self, args: str = "") -> None:
        """Print working directory (like Unix pwd)."""
        self._print(os.getcwd())

    def _cmd_mkdir(self, args: str = "") -> None:
        """Create directories (like Unix mkdir). Supports -p (parent)."""
        parts = args.strip().split()
        flags = {p for p in parts if p.startswith("-")}
        targets = [p for p in parts if not p.startswith("-")]
        parent = "-p" in flags
        if not targets:
            self._print("  Usage: mkdir [-p] <dir> ...")
            self._last_exit_code = 1
            return
        for d in targets:
            path = os.path.expanduser(d)
            try:
                if parent:
                    os.makedirs(path, exist_ok=True)
                else:
                    os.mkdir(path)
            except FileExistsError:
                self._print(f"  mkdir: cannot create directory '{d}': File exists")
                self._last_exit_code = 1
            except FileNotFoundError:
                self._print(f"  mkdir: cannot create directory '{d}': No such file or directory")
                self._last_exit_code = 1
            except PermissionError:
                self._print(f"  mkdir: permission denied: {d}")
                self._last_exit_code = 1

    def _cmd_rm(self, args: str = "") -> None:
        """Remove files or directories (like Unix rm). Supports -r, -f."""
        parts = args.strip().split()
        flags = {p for p in parts if p.startswith("-")}
        targets = [p for p in parts if not p.startswith("-")]
        recursive = "-r" in flags or "-rf" in flags or "-fr" in flags
        force = "-f" in flags or "-rf" in flags or "-fr" in flags
        if not targets:
            self._print("  Usage: rm [-r] [-f] <path> ...")
            self._last_exit_code = 1
            return
        for t in targets:
            path = os.path.expanduser(t)
            try:
                if os.path.isdir(path) and not recursive:
                    self._print(f"  rm: cannot remove '{t}': Is a directory")
                    self._last_exit_code = 1
                    continue
                if recursive and os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
            except FileNotFoundError:
                if not force:
                    self._print(f"  rm: cannot remove '{t}': No such file or directory")
                    self._last_exit_code = 1
            except PermissionError:
                self._print(f"  rm: permission denied: {t}")
                self._last_exit_code = 1

    def _cmd_cp(self, args: str = "") -> None:
        """Copy files (like Unix cp). Supports -r for directories."""
        parts = args.strip().split()
        flags = {p for p in parts if p.startswith("-")}
        targets = [p for p in parts if not p.startswith("-")]
        recursive = "-r" in flags or "-R" in flags
        if len(targets) < 2:
            self._print("  Usage: cp [-r] <source> <dest>")
            self._last_exit_code = 1
            return
        src, dst = os.path.expanduser(targets[0]), os.path.expanduser(targets[1])
        try:
            if os.path.isdir(src):
                if recursive:
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    self._print(f"  cp: omitting directory '{targets[0]}'")
                    self._last_exit_code = 1
            else:
                shutil.copy2(src, dst)
        except FileNotFoundError:
            self._print(f"  cp: cannot stat '{targets[0]}': No such file or directory")
            self._last_exit_code = 1
        except PermissionError:
            self._print(f"  cp: permission denied")
            self._last_exit_code = 1
        except IsADirectoryError:
            self._print(f"  cp: '{dst}' is a directory (use -r)")
            self._last_exit_code = 1

    def _cmd_mv(self, args: str = "") -> None:
        """Move/rename files (like Unix mv)."""
        parts = args.strip().split()
        targets = [p for p in parts if not p.startswith("-")]
        if len(targets) < 2:
            self._print("  Usage: mv <source> <dest>")
            self._last_exit_code = 1
            return
        src, dst = os.path.expanduser(targets[0]), os.path.expanduser(targets[1])
        try:
            shutil.move(src, dst)
        except FileNotFoundError:
            self._print(f"  mv: cannot stat '{targets[0]}': No such file or directory")
            self._last_exit_code = 1
        except PermissionError:
            self._print(f"  mv: permission denied")
            self._last_exit_code = 1

    def _cmd_cat(self, args: str = "") -> None:
        """Concatenate and print files (like Unix cat). Also reads AI devices (/dev/llm, etc)."""
        if not args:
            if self._piped_input:
                self._print(self._piped_input, end="")
            else:
                self._print("  Usage: cat <file> ...   or   <cmd> | cat")
                self._last_exit_code = 1
            return
        parts = args.strip().split()
        first = parts[0]
        first_path = os.path.expanduser(first)
        vfs = self.os.vfs
        if vfs and (first_path.startswith("/dev/") or first_path.startswith("/proc/")):
            remaining = " ".join(parts[1:]) if len(parts) > 1 else ""
            content = vfs.read(first_path)
            if content is not None:
                self._print(content, end="\n" if not content.endswith("\n") else "")
            else:
                self._print(f"  cat: {first}: No such file or directory")
                self._last_exit_code = 1
            return
        for f in parts:
            path = os.path.expanduser(f)
            try:
                with open(path) as fp:
                    self._print(fp.read(), end="")
            except FileNotFoundError:
                self._print(f"  cat: {f}: No such file or directory")
                self._last_exit_code = 1
            except PermissionError:
                self._print(f"  cat: {f}: Permission denied")
                self._last_exit_code = 1
            except IsADirectoryError:
                self._print(f"  cat: {f}: Is a directory")
                self._last_exit_code = 1

    def _cmd_lsdev(self, args: str = "") -> None:
        """List AI device nodes (/dev/*)."""
        if not self.os.devices:
            self._print("  Devices not available (not booted?)")
            return
        self._print("  AI Device nodes:")
        self._print(self.os.devices.list_devices())

    def _cmd_touch(self, args: str = "") -> None:
        """Update file timestamps or create empty files (like Unix touch)."""
        if not args:
            self._print("  Usage: touch <file> ...")
            self._last_exit_code = 1
            return
        parts = args.strip().split()
        for f in parts:
            path = os.path.expanduser(f)
            try:
                if os.path.exists(path):
                    os.utime(path, None)
                else:
                    open(path, 'a').close()
            except PermissionError:
                self._print(f"  touch: {f}: Permission denied")
                self._last_exit_code = 1

    def _cmd_chmod(self, args: str = "") -> None:
        """Change file mode (like Unix chmod). Supports octal (755) and symbolic (u+x)."""
        parts = args.strip().split()
        if len(parts) < 2:
            self._print("  Usage: chmod <mode> <file> ...")
            self._last_exit_code = 1
            return
        mode_str = parts[0]
        targets = parts[1:]
        for t in targets:
            path = os.path.expanduser(t)
            try:
                if mode_str.isdigit():
                    mode = int(mode_str, 8)
                    os.chmod(path, mode)
                else:
                    # Simple symbolic: u+x, g+w, o-r, a+x
                    current = os.stat(path).st_mode
                    who = mode_str[0] if mode_str[0] in 'ugoa' else 'a'
                    op = mode_str[1] if len(mode_str) > 1 and mode_str[1] in '+-=' else '+'
                    perm = mode_str[2:] if len(mode_str) > 2 else ''
                    who_map = {'u': stat.S_IRWXU, 'g': stat.S_IRWXG, 'o': stat.S_IRWXO, 'a': stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO}
                    perm_map = {'x': stat.S_IXUSR, 'w': stat.S_IWUSR, 'r': stat.S_IRUSR}
                    mask = 0
                    for p in perm:
                        mask |= perm_map.get(p, 0)
                    if who != 'a':
                        # Restrict mask to only the specified who bits
                        mask = mask & who_map.get(who, 0o777)
                        # Also shift to group/other
                        if who == 'g':
                            mask = (mask >> 3) & 0o777
                        elif who == 'o':
                            mask = (mask >> 6) & 0o777
                    if op == '+':
                        new_mode = current | mask
                    elif op == '-':
                        new_mode = current & ~mask
                    else:
                        new_mode = mask
                    os.chmod(path, new_mode)
            except FileNotFoundError:
                self._print(f"  chmod: cannot access '{t}': No such file or directory")
                self._last_exit_code = 1
            except PermissionError:
                self._print(f"  chmod: changing permissions of '{t}': Operation not permitted")
                self._last_exit_code = 1

    def _cmd_find(self, args: str = "") -> None:
        """Recursively find files matching a pattern (like Unix find). Usage: find [path] [-name <glob>]"""
        if not args.strip():
            self._print("  Usage: find [path] [-name <glob>]")
            self._last_exit_code = 1
            return
        try:
            import shlex
            parts = shlex.split(args)
        except Exception:
            parts = args.strip().split()
        path = "."
        name_pat = "*"
        i = 0
        while i < len(parts):
            if parts[i] == "-name" and i + 1 < len(parts):
                name_pat = parts[i + 1].strip("'\"")
                i += 2
            elif parts[i] == "-iname" and i + 1 < len(parts):
                name_pat = parts[i + 1].strip("'\"")
                i += 2
            else:
                path = parts[i]
                i += 1
        root = os.path.expanduser(path)
        if not os.path.isdir(root):
            self._print(f"  find: '{path}': No such directory")
            self._last_exit_code = 1
            return
        count = 0
        for dirpath, dirnames, filenames in os.walk(root):
            for f in filenames:
                if fnmatch.fnmatch(f, name_pat):
                    self._print(f"  {os.path.join(dirpath, f)}")
                    count += 1
        self._print(f"  [{count} file(s) found]")

    # ── Unix scripting commands ────────────────────────────────────

    def _cmd_true(self, args: str = "") -> None:
        """Return exit code 0 (like Unix true)."""
        self._last_exit_code = 0

    def _cmd_false(self, args: str = "") -> None:
        """Return exit code 1 (like Unix false)."""
        self._last_exit_code = 1

    def _cmd_test(self, args: str = "") -> None:
        """Evaluate expression (like Unix test / [). Supports -f, -d, -e, -n, -z, =, !=."""
        if not args:
            self._last_exit_code = 1
            return
        parts = args.strip().split()
        if parts[0] == "[":
            parts = parts[1:]
        if parts and parts[-1] == "]":
            parts = parts[:-1]
        if not parts:
            self._last_exit_code = 1
            return

        def _file_test(flag: str, path: str) -> bool:
            p = os.path.expanduser(path)
            if flag == "-e": return os.path.exists(p)
            if flag == "-f": return os.path.isfile(p)
            if flag == "-d": return os.path.isdir(p)
            if flag == "-r": return os.access(p, os.R_OK)
            if flag == "-w": return os.access(p, os.W_OK)
            if flag == "-x": return os.access(p, os.X_OK)
            if flag == "-s": return os.path.isfile(p) and os.path.getsize(p) > 0
            if flag == "-L": return os.path.islink(p)
            return False

        if len(parts) == 1:
            self._last_exit_code = 0 if parts[0] else 1
        elif len(parts) == 2:
            self._last_exit_code = 0 if _file_test(parts[0], parts[1]) else 1
        elif len(parts) == 3:
            a, op, b = parts[0], parts[1], parts[2]
            if op == "=":
                self._last_exit_code = 0 if a == b else 1
            elif op == "!=":
                self._last_exit_code = 0 if a != b else 1
            elif op == "-eq":
                try: self._last_exit_code = 0 if int(a) == int(b) else 1
                except (ValueError, TypeError): self._last_exit_code = 1
            elif op == "-ne":
                try: self._last_exit_code = 0 if int(a) != int(b) else 1
                except (ValueError, TypeError): self._last_exit_code = 1
            elif op == "-gt":
                try: self._last_exit_code = 0 if int(a) > int(b) else 1
                except (ValueError, TypeError): self._last_exit_code = 1
            elif op == "-lt":
                try: self._last_exit_code = 0 if int(a) < int(b) else 1
                except (ValueError, TypeError): self._last_exit_code = 1
            elif op == "-ge":
                try: self._last_exit_code = 0 if int(a) >= int(b) else 1
                except (ValueError, TypeError): self._last_exit_code = 1
            elif op == "-le":
                try: self._last_exit_code = 0 if int(a) <= int(b) else 1
                except (ValueError, TypeError): self._last_exit_code = 1
            elif op == "-n":
                self._last_exit_code = 0 if len(a) > 0 else 1
            elif op == "-z":
                self._last_exit_code = 0 if len(a) == 0 else 1
            else:
                self._last_exit_code = 1
        else:
            self._last_exit_code = 1

    def _cmd_which(self, args: str = "") -> None:
        """Locate a command (like Unix which)."""
        if not args:
            self._print("  Usage: which <command>")
            self._last_exit_code = 1
            return
        cmd = args.strip().lower()
        if cmd in self.COMMANDS or cmd in self._aliases:
            if cmd in self._aliases:
                self._print(f"  {cmd}: aliased to {self._aliases[cmd]}")
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
        elif cmd in self.COMMANDS:
            self._print(f"  {cmd} is a shell built-in")
        elif shutil.which(cmd):
            self._print(f"  {cmd} is {shutil.which(cmd)}")
        else:
            self._print(f"  {cmd}: not found")
            self._last_exit_code = 1

    def _cmd_env(self, args: str = "") -> None:
        """Print environment variables (like Unix env)."""
        max_key = max(len(k) for k in self._env) if self._env else 0
        for k in sorted(self._env):
            self._print(f"  {k.upper().ljust(max_key)}  {self._env[k]}")

    def _cmd_font(self, args: str = "") -> None:
        """Set or show terminal font (iTerm2). Uses OSC 50 escape sequence."""
        if not args:
            current = self._env.get("FONT", "terminal default")
            self._print(f"  Current font: {current}")
            self._print("  Usage: font <name>   (e.g. 'font Digital TS Medium')")
            return
        font_name = args.strip()
        self._env["FONT"] = font_name
        self.state.set_env("FONT", font_name)
        self.state.save()
        # Emit OSC 50 to change terminal font (iTerm2, some terminals)
        sys.stdout.write(f"\x1b]50;Set Font={font_name}\x07")
        sys.stdout.flush()
        self._print(f"  Font set to: {font_name}")

    def _cmd_export_state(self, args: str = "") -> None:
        self._print(self._dump_json(self.state.to_dict()))

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
            value = input(f"  {prompt}").strip()
            self._env[parts[0]] = value
            self._last_exit_code = 0
        except (EOFError, KeyboardInterrupt):
            self._last_exit_code = 1

    def _cmd_printf(self, args: str = "") -> None:
        """Format and print data (like bash printf). Supports %s, %d, %f."""
        if not args:
            self._last_exit_code = 1
            return
        parts = args.split(maxsplit=1)
        fmt = parts[0]
        values = parts[1].split() if len(parts) > 1 else []
        try:
            idx = 0
            result = []
            i = 0
            while i < len(fmt):
                if fmt[i] == '%' and i + 1 < len(fmt):
                    spec = fmt[i + 1]
                    if spec in ('s', 'd', 'f', '%'):
                        if spec == '%':
                            result.append('%')
                        elif idx < len(values):
                            if spec == 's':
                                result.append(values[idx])
                            elif spec == 'd':
                                result.append(str(int(float(values[idx]))))
                            elif spec == 'f':
                                result.append(f"{float(values[idx]):f}")
                            idx += 1
                        i += 2
                        continue
                    result.append(fmt[i])
                elif fmt[i] == '\\' and i + 1 < len(fmt):
                    esc = {'n': '\n', 't': '\t', 'r': '\r', '\\': '\\'}.get(fmt[i+1], fmt[i+1])
                    result.append(esc)
                    i += 2
                    continue
                else:
                    result.append(fmt[i])
                i += 1
            self._print("".join(result), end="")
        except (ValueError, IndexError):
            self._last_exit_code = 1

    def _cmd_dirname(self, args: str = "") -> None:
        """Strip last component from path (like Unix dirname)."""
        if not args:
            self._print("  Usage: dirname <path>")
            self._last_exit_code = 1
            return
        self._print(f"  {os.path.dirname(os.path.expanduser(args.strip()))}")

    def _cmd_basename(self, args: str = "") -> None:
        """Strip directory from path (like Unix basename)."""
        if not args:
            self._print("  Usage: basename <path>")
            self._last_exit_code = 1
            return
        self._print(f"  {os.path.basename(os.path.expanduser(args.strip()))}")

    def _cmd_yes(self, args: str = "") -> None:
        """Repeatedly output a string (like Unix yes). Ctrl+C to stop."""
        string = args.strip() or "y"
        try:
            while True:
                self._print(string)
        except KeyboardInterrupt:
            pass

    def _cmd_xargs(self, args: str = "") -> None:
        """Build and execute command lines from piped input (like Unix xargs).
        Reads piped input, splits by whitespace, and runs the given command
        with each item as an argument."""
        data = self._piped_input if self._piped_input else ""
        if not data or not args:
            self._print("  Usage: <cmd> | xargs <command>")
            self._last_exit_code = 1
            return
        parts = args.strip().split()
        cmd = parts[0]
        base_args = parts[1:] if len(parts) > 1 else []
        items = data.split()
        for item in items:
            all_args = base_args + [item]
            raw = cmd + " " + " ".join(all_args)
            out = self._execute_single(raw, "")
            self._print(out, end="")

    def _cmd_cut(self, args: str = "") -> None:
        """Remove sections from each line of input (like Unix cut).
        Usage: <cmd> | cut -d<delim> -f<field>"""
        data = self._piped_input if self._piped_input else ""
        if not data:
            data = sys.stdin.read()
        if not data:
            self._print("  Usage: <cmd> | cut -d<delim> -f<field>")
            self._last_exit_code = 1
            return
        parts = args.strip().split()
        delim = "\t"
        fields = [1]
        for i, p in enumerate(parts):
            if p.startswith("-d") and len(p) > 2:
                delim = p[2]
            elif p.startswith("-f"):
                fspec = p[2:]
                if fspec:
                    fields = [int(x) for x in fspec.split(",")]
        result = []
        for line in data.split("\n"):
            if not line.strip():
                result.append("")
                continue
            cols = line.split(delim)
            selected = [cols[f-1] for f in fields if f <= len(cols)]
            result.append(delim.join(selected))
        self._print("\n".join(result))

    def _cmd_tr(self, args: str = "") -> None:
        """Translate or delete characters (like Unix tr).
        Usage: <cmd> | tr <set1> <set2>   or   <cmd> | tr -d <set>"""
        data = self._piped_input if self._piped_input else ""
        if not data:
            self._print("  Usage: <cmd> | tr <set1> <set2>")
            self._last_exit_code = 1
            return
        parts = args.strip().split()
        if not parts:
            self._print("  Usage: <cmd> | tr <set1> <set2>")
            self._last_exit_code = 1
            return
        delete_mode = parts[0] == "-d"
        if delete_mode:
            if len(parts) < 2:
                return
            chars = parts[1]
            trans_table = str.maketrans("", "", chars)
        else:
            if len(parts) < 2:
                return
            set1 = parts[0]
            set2 = parts[1] if len(parts) > 1 else set1
            trans_table = str.maketrans(set1, set2)
        self._print(data.translate(trans_table), end="")

    def _cmd_pbcopy(self, args: str = "") -> None:
        """Copy piped input to macOS clipboard (like pbcopy). Usage: <cmd> | pbcopy"""
        data = self._piped_input if self._piped_input else args
        if not data:
            self._print("  Usage: <cmd> | pbcopy   or   pbcopy <text>")
            self._last_exit_code = 1
            return
        try:
            p = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
            p.communicate(data.encode("utf-8"))
            self._print(f"  Copied {len(data)} chars to clipboard")
        except Exception as e:
            self._print(f"  pbcopy: {e}")
            self._last_exit_code = 1

    def _cmd_pbpaste(self, args: str = "") -> None:
        """Paste from macOS clipboard (like pbpaste)."""
        try:
            p = subprocess.run(["pbpaste"], capture_output=True)
            self._print(p.stdout.decode("utf-8").rstrip())
        except Exception as e:
            self._print(f"  pbpaste: {e}")
            self._last_exit_code = 1

    # ── Init / Boot commands ────────────────────────────────────────

    def _cmd_boot(self, args: str = "") -> None:
        """Boot the shell — start kernel + init system + services."""
        if self._running and self._piped_input is None:
            self._print("  Already booted. Use 'shutdown' to halt, then 'sloughgpt shell' to restart.")
            return
        self._running = True
        log = self.os.boot(shell_run=self._shell_cmd if hasattr(self, "_shell_cmd") else None)
        self._print(log)

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
            self._print(f"  Current runlevel: {init._current_runlevel}")

        else:
            self._print("  Usage: svc [list|start|stop|restart|status] [name]")
            self._last_exit_code = 1

    def _cmd_asm(self, args: str = "") -> None:
        """Assemble and run a VM program. Usage: asm <file.asm>   or   piped | asm"""
        source = self._piped_input if self._piped_input else ""
        file_path = args.strip() if args else ""

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

        if args.strip() == "--test" or args.strip() == "--self-test":
            from domains.shell.vm import self_test as _vm_self_test
            results = _vm_self_test()
            self._print("  VM Self-Test:")
            for line in results:
                self._print(line)
            return

        if args.strip() == "--list" or args.strip() == "-l":
            self._print("  Built-in programs (use asm --test to run):")
            self._print("    hello      Hello World")
            self._print("    counter    Count 0..9")
            self._print("    fib        Fibonacci 0..12")
            self._print("    collatz    Collatz from 27")
            return

        try:
            from domains.shell.vm import VMRunner, VMFault
            runner = VMRunner(device_manager=self.os.devices)
            output = runner.assemble_and_run(source)
            for line in output:
                self._print(line)
        except VMFault as e:
            self._print(f"  Assembly error: {e}")
            self._last_exit_code = 1
        except Exception as e:
            self._print(f"  VM error: {e}")
            self._last_exit_code = 1

    # ── Command registry ────────────────────────────────────────────

    COMMANDS: dict[str, Callable] = {
        "help": _cmd_help,
        "clear": _cmd_clear,
        "exit": _cmd_exit,
        "ls": _cmd_ls,
        "cd": _cmd_cd,
        "pwd": _cmd_pwd,
        "mkdir": _cmd_mkdir,
        "rm": _cmd_rm,
        "cp": _cmd_cp,
        "mv": _cmd_mv,
        "cat": _cmd_cat,
        "touch": _cmd_touch,
        "chmod": _cmd_chmod,
        "true": _cmd_true,
        "false": _cmd_false,
        "test": _cmd_test,
        "[": _cmd_test,
        "which": _cmd_which,
        "type": _cmd_type,
        "env": _cmd_env,
        "font": _cmd_font,
        "history": _cmd_history,
        "fc": _cmd_fc,
        "alias": _cmd_alias,
        "unalias": _cmd_unalias,
        "export": _cmd_export,
        "set": _cmd_set,
        "source": _cmd_source,
        ".": _cmd_source,
        "py": _cmd_py,
        "pushd": _cmd_pushd,
        "popd": _cmd_popd,
        "dirs": _cmd_dirs,
        "procs": _cmd_procs,
        "kill": _cmd_kill,
        "bg": _cmd_bg,
        "jobs": _cmd_bg,
        "fg": _cmd_fg,
        "sleep": _cmd_sleep,
        "watch": _cmd_watch,
        "models": _cmd_models,
        "load": _cmd_load,
        "unload": _cmd_unload,
        "souls": _cmd_souls,
        "switch": _cmd_switch,
        "whoami": _cmd_whoami,
        "uname": _cmd_uname,
        "uptime": _cmd_uptime,
        "health": _cmd_health,
        "status": _cmd_status,
        "metrics": _cmd_metrics,
        "datasets": _cmd_datasets,
        "knowledge": _cmd_knowledge,
        "checkpoints": _cmd_checkpoints,
        "finetuned": _cmd_finetuned,
        "train": _cmd_train,
        "gen": _cmd_gen,
        "chat": _cmd_chat,
        "tokenizer": _cmd_tokenizer,
        "ai": _cmd_ai,
        "agents": _cmd_agents,
        "tutorial": _cmd_tutorial,
        "grep": _cmd_grep,
        "head": _cmd_head,
        "tail": _cmd_tail,
        "wc": _cmd_wc,
        "tee": _cmd_tee,
        "sort": _cmd_sort,
        "uniq": _cmd_uniq,
        "less": _cmd_less,
        "echo": _cmd_echo,
        "read": _cmd_read,
        "printf": _cmd_printf,
        "dirname": _cmd_dirname,
        "basename": _cmd_basename,
        "yes": _cmd_yes,
        "find": _cmd_find,
        "xargs": _cmd_xargs,
        "cut": _cmd_cut,
        "tr": _cmd_tr,
        "wm": _cmd_wm,
        "win": _cmd_wm,
        "pbcopy": _cmd_pbcopy,
        "pbpaste": _cmd_pbpaste,
        "remember": _cmd_remember,
        "recall": _cmd_recall,
        "boot": _cmd_boot,
        "shutdown": _cmd_shutdown,
        "svc": _cmd_svc,
        "devices": _cmd_lsdev,
        "lsdev": _cmd_lsdev,
        "asm": _cmd_asm,
    }

    # ── Main loop ───────────────────────────────────────────────────

    def run(self) -> None:
        boot_log = self.os.boot()
        self._print(boot_log)
        self._running = True
        self._print_header()
        if self.state.first_run:
            self._show_welcome()
        while self._running:
            try:
                prompt = self._render_prompt()
                line = input(f" {prompt} ").strip()
            except EOFError:
                self._print()
                break
            except KeyboardInterrupt:
                self._print("^C")
                continue

            # Multiline continuation with trailing backslash
            while line.endswith("\\") and not line.endswith("\\\\"):
                line = line.rstrip("\\").rstrip()
                try:
                    continuation = input("  > ").strip()
                    line = f"{line} {continuation}"
                except (EOFError, KeyboardInterrupt):
                    break

            if not line:
                continue

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
                elif len(commands) > 1:
                    self._execute_pipeline(commands, should_time=should_time)
                else:
                    raw_cmd, op = commands[0]
                    expanded = self._expand_alias(raw_cmd)
                    parts = expanded.split(maxsplit=1)
                    cmd = parts[0].lower()
                    args = parts[1] if len(parts) > 1 else ""
                    handler = self.COMMANDS.get(cmd)
                    if handler:
                        if should_time:
                            import time as _time
                            t0 = _time.time()
                            try:
                                handler(self, args)
                                self._last_exit_code = 0
                            except SystemExit as e:
                                self._last_exit_code = e.code if isinstance(e.code, int) else 1
                            except Exception as e:
                                self._print(f"  {_C_RED}Error:{_C_RESET} {e}")
                                self._last_exit_code = 1
                            elapsed = _time.time() - t0
                            self._print(f"{_C_DIM}  [{elapsed:.2f}s]{_C_RESET}")
                        else:
                            try:
                                handler(self, args)
                                self._last_exit_code = 0
                            except SystemExit as e:
                                self._last_exit_code = e.code if isinstance(e.code, int) else 1
                            except Exception as e:
                                self._print(f"  {_C_RED}Error:{_C_RESET} {e}")
                                self._last_exit_code = 1
                    else:
                        suggestion = self._suggest_command(cmd)
                        msg = f"  {_C_RED}Unknown command:{_C_RESET} {cmd}. Type `help`."
                        if suggestion:
                            msg += f" Did you mean `{_C_YELLOW}{suggestion}{_C_RESET}`?"
                        self._print(msg)
                        self._last_exit_code = 127
            except KeyboardInterrupt:
                self._print(f"  {_C_DIM}Aborted{_C_RESET}")
                self._aborted = True
            except Exception as e:
                self._print(f"  {_C_RED}Error:{_C_RESET} {e}")

        self.state.save()
        self.os.shutdown()
