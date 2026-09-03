"""
mini_cli — Lightweight CLI framework with auto-correct, fuzzy matching,
and grouped help. Replaces Click for sloughgpt.

Features:
- Command groups with subcommands
- Options with short aliases, flags, defaults
- Positional arguments (required, optional, variadic)
- Context passing (host, port, json, etc.)
- Auto-correct: "sloughgpt md" → "Did you mean model? [Y/n]"
- Frequency-ranked fuzzy matching
- Grouped, color-coded help output
- No external dependencies beyond stdlib
"""

import os
import sys
import json
import shlex
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# ── ANSI ────────────────────────────────────────────────────────────────

_TTY = sys.stdout.isatty()

def _c(text: str, code: str) -> str:
    return f"{code}{text}\033[0m" if _TTY else text

_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"


# ── Usage tracking ──────────────────────────────────────────────────────

_USAGE_PATH = Path.home() / ".config" / "sloughgpt" / "usage_stats.json"

def _load_usage() -> dict:
    try:
        if _USAGE_PATH.exists():
            return json.loads(_USAGE_PATH.read_text())
    except Exception:
        pass
    return {}

def _save_usage(data: dict) -> None:
    try:
        _USAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _USAGE_PATH.write_text(json.dumps(data, indent=2))
    except Exception:
        pass

def _record_usage(cmd_name: str) -> None:
    usage = _load_usage()
    usage[cmd_name] = usage.get(cmd_name, 0) + 1
    _save_usage(usage)


# ── Choice type ─────────────────────────────────────────────────────────

class Choice:
    """Constrained value type (like click.Choice)."""

    def __init__(self, choices: List[str], case_sensitive: bool = False):
        self.choices = choices
        self.case_sensitive = case_sensitive

    def convert(self, value: str, param_name: str) -> str:
        if self.case_sensitive:
            if value in self.choices:
                return value
        else:
            for c in self.choices:
                if c.lower() == value.lower():
                    return c
        valid = ", ".join(self.choices)
        raise ValueError(f"Invalid value '{value}' for {param_name}. Choose from: {valid}")


# ── Parameter definitions ───────────────────────────────────────────────

class Option:
    """An option like --name VALUE or --flag."""

    def __init__(self, names: List[str], help: str = "", default: Any = None,
                 type: type = str, is_flag: bool = False, required: bool = False,
                 multiple: bool = False, metavar: str = "", show_default: bool = False,
                 choice: Optional[Choice] = None):
        self.names = names
        self.help = help
        self.default = default
        self.type = type
        self.is_flag = is_flag
        self.required = required
        self.multiple = multiple
        self.metavar = metavar
        self.show_default = show_default
        self.choice = choice
        # The param name (without --) used as the key
        self.dest = names[0].lstrip("-").replace("-", "_")

    @property
    def primary(self) -> str:
        return self.names[0]

    @property
    def short(self) -> Optional[str]:
        for n in self.names:
            if len(n) == 2 and n.startswith("-"):
                return n
        return None


class Argument:
    """A positional argument."""

    def __init__(self, name: str, required: bool = True, default: Any = None,
                 nargs: int = 1):
        self.name = name
        self.required = required
        self.default = default
        self.nargs = nargs  # -1 = variadic


# ── Command ─────────────────────────────────────────────────────────────

class Command:
    """A single command (leaf in the tree)."""

    def __init__(self, name: str, func: Callable, help: str = "",
                 options: Optional[List[Option]] = None,
                 arguments: Optional[List[Argument]] = None,
                 hidden: bool = False):
        self.name = name
        self.func = func
        self.help = help
        self.options = options or []
        self.arguments = arguments or []
        self.hidden = hidden


# ── Group ───────────────────────────────────────────────────────────────

class Group:
    """A command group with subcommands. Supports auto-correct."""

    def __init__(self, name: str = "", help: str = "",
                 invoke_without_command: bool = False,
                 options: Optional[List[Option]] = None):
        self.name = name
        self.help = help
        self.commands: Dict[str, Command] = {}
        self.groups: Dict[str, "Group"] = {}
        self.parent: Optional["Group"] = None
        self.invoke_without_command = invoke_without_command
        self.options = options or []

    def command(self, name: str = "", help: str = "", hidden: bool = False):
        """Decorator to register a command."""
        def decorator(func):
            cmd_name = name or func.__name__
            cmd = Command(cmd_name, func, help, hidden=hidden)
            # Extract options/arguments from func metadata
            cmd.options = getattr(func, "_options", [])
            cmd.arguments = getattr(func, "_arguments", [])
            self.commands[cmd_name] = cmd
            return func
        return decorator

    def group(self, name: str = "", help: str = ""):
        """Decorator to register a subgroup."""
        def decorator(func):
            grp_name = name or func.__name__
            grp = Group(grp_name, help)
            func._group = grp
            self.groups[grp_name] = grp
            return func
        return decorator

    def add_command(self, cmd: Command, name: str = ""):
        """Programmatically add a command."""
        self.commands[name or cmd.name] = cmd

    def add_group(self, grp: "Group", name: str = ""):
        """Programmatically add a group."""
        self.groups[name or grp.name] = grp

    def _fuzzy_match(self, cmd_name: str) -> List[str]:
        """Find close matches for a command name."""
        all_names = list(self.commands.keys()) + list(self.groups.keys())

        # Prefix match
        prefix = [c for c in all_names if c.startswith(cmd_name.lower())]
        if prefix:
            return prefix

        # Substring match
        substring = [c for c in all_names if cmd_name.lower() in c.lower()]
        if substring:
            return substring

        # Fuzzy match
        cutoff = 0.4 if len(cmd_name) <= 3 else 0.6
        return get_close_matches(cmd_name, all_names, n=3, cutoff=cutoff)


# ── Context ─────────────────────────────────────────────────────────────

class Context:
    """Shared context passed to commands."""

    def __init__(self, obj: Optional[dict] = None):
        self.obj = obj or {}
        self.invoked_subcommand: Optional[str] = None
        self._group: Optional[Group] = None

    def ensure_object(self, factory: Callable = dict):
        if not self.obj:
            self.obj = factory()


# ── Parser ──────────────────────────────────────────────────────────────

def _parse_args(args: List[str], options: List[Option], arguments: List[Argument]
               ) -> Tuple[dict, List[str]]:
    """Parse raw args into (kwargs, positional_args)."""
    kwargs = {}
    positional = []
    i = 0

    # Set defaults for flags
    for opt in options:
        if opt.is_flag:
            kwargs[opt.dest] = False
        elif opt.multiple:
            kwargs[opt.dest] = []
        elif opt.default is not None:
            kwargs[opt.dest] = opt.default

    while i < len(args):
        arg = args[i]

        if arg.startswith("-"):
            # Option
            matched = False
            for opt in options:
                if arg in opt.names:
                    if opt.is_flag:
                        # Check for --no- prefix
                        if arg.startswith("--no-"):
                            kwargs[opt.dest] = False
                        else:
                            kwargs[opt.dest] = True
                        matched = True
                        break
                    else:
                        # Need a value
                        i += 1
                        if i >= len(args):
                            raise ValueError(f"Option {arg} requires a value")
                        value = args[i]
                        if opt.choice:
                            value = opt.choice.convert(value, opt.primary)
                        if opt.type == int:
                            value = int(value)
                        elif opt.type == float:
                            value = float(value)
                        if opt.multiple:
                            kwargs.setdefault(opt.dest, []).append(value)
                        else:
                            kwargs[opt.dest] = value
                        matched = True
                        break

            if not matched:
                # Check for --key=VALUE style
                if "=" in arg:
                    key, value = arg.split("=", 1)
                    for opt in options:
                        if key in opt.names:
                            if opt.choice:
                                value = opt.choice.convert(value, opt.primary)
                            if opt.type == int:
                                value = int(value)
                            elif opt.type == float:
                                value = float(value)
                            kwargs[opt.dest] = value
                            matched = True
                            break

                if not matched:
                    raise ValueError(f"Unknown option: {arg}")
        else:
            positional.append(arg)

        i += 1

    # Validate required options
    for opt in options:
        if opt.required and opt.dest not in kwargs:
            raise ValueError(f"Missing required option: {opt.primary}")

    # Map positional args to arguments
    pos_idx = 0
    for arg_def in arguments:
        if arg_def.nargs == -1:
            # Variadic — take everything remaining
            kwargs[arg_def.name] = positional[pos_idx:]
            pos_idx = len(positional)
        elif pos_idx < len(positional):
            kwargs[arg_def.name] = positional[pos_idx]
            pos_idx += 1
        elif arg_def.required:
            raise ValueError(f"Missing required argument: {arg_def.name}")
        else:
            kwargs[arg_def.name] = arg_def.default

    return kwargs, positional[pos_idx:]


# ── Help formatting ─────────────────────────────────────────────────────

def _format_help(group: Group, ctx: Context) -> str:
    """Generate grouped, color-coded help text."""
    lines = []

    # Header
    try:
        from core.version import format_version_display
        version = format_version_display()
    except Exception:
        version = "dev"

    lines.append(f"\n  {_c('SloughGPT', _BOLD + _CYAN)} {_c(f'({version})', _DIM)}\n")
    lines.append(f"  {_c('Train, chat, serve, and manage AI models', _DIM)}\n")

    # Commands grouped by category
    _p()
    _p(f"  {_c('Commands:', _BOLD)}")

    printed = set()

    for cat_name, cat_info in _CATEGORIES.items():
        cat_cmds = []
        for cmd_name in cat_info["cmds"]:
            if cmd_name in group.commands:
                cat_cmds.append(("cmd", group.commands[cmd_name]))
            elif cmd_name in group.groups:
                cat_cmds.append(("grp", group.groups[cmd_name]))
        if cat_cmds:
            desc = cat_info["desc"]
            _p(f"\n    {_c(cat_name, _BOLD + _YELLOW)} {_c(f'— {desc}', _DIM)}")
            for kind, item in cat_cmds:
                if kind == "cmd":
                    if item.hidden:
                        continue
                    padded = item.name.ljust(16)
                    help_text = item.help[:50]
                    _p(f"      {_c(padded, _CYAN)} {_c(help_text, _DIM)}")
                else:
                    padded = item.name.ljust(16)
                    help_text = item.help[:50]
                    _p(f"      {_c(padded, _CYAN)} {_c(help_text, _DIM)}")
                printed.add(item.name)

    # Remaining commands
    remaining = []
    for name in sorted(group.commands.keys()):
        if name not in printed and not group.commands[name].hidden:
            remaining.append(("cmd", group.commands[name]))
    for name in sorted(group.groups.keys()):
        if name not in printed:
            remaining.append(("grp", group.groups[name]))

    if remaining:
        _p(f"\n    {_c('Other', _BOLD + _YELLOW)}")
        for kind, item in remaining:
            padded = item.name.ljust(16)
            help_text = item.help[:50]
            _p(f"      {_c(padded, _CYAN)} {_c(help_text, _DIM)}")

    # Global options
    _p(f"\n  {_c('Global Options:', _BOLD)}")
    _p(f"    {_c('--host', _CYAN)}         API hostname (default: localhost)")
    _p(f"    {_c('--port', _CYAN)}         API port (default: 8000)")
    _p(f"    {_c('-c, --config', _CYAN)}   Config path (default: config.yaml)")
    _p(f"    {_c('--json', _CYAN)}         JSON output for commands")
    _p(f"    {_c('--no-color', _CYAN)}     Disable ANSI color output")
    _p(f"    {_c('-q, --quiet', _CYAN)}    Suppress non-essential output")
    _p(f"    {_c('--timeout', _CYAN)}      HTTP timeout in seconds (default: 10)")
    _p(f"    {_c('--version', _CYAN)}      Show version")
    _p(f"    {_c('--yes, -y', _CYAN)}      Skip all confirmations")
    _p(f"    {_c('--help', _CYAN)}         Show this help message")
    _p()

    # Tips
    _p(f"  {_c('Examples:', _BOLD)}")
    _p(f"    sloughgpt chat                     Start chatting")
    _p(f"    sloughgpt model download gpt2     Download a model")
    _p(f"    sloughgpt model status             Check model cache")
    _p(f"    sloughgpt train dataset shakespeare Train on dataset")
    _p(f"    sloughgpt shell                    Interactive shell")
    _p()

    _p(f"  {_c('Tips:', _BOLD)}")
    _p(f"    {_c('•', _GREEN)} Use fuzzy matching — {_c("'sloughgpt md'", _CYAN)} finds {_c('model', _CYAN)}")
    _p(f"    {_c('•', _GREEN)} Run {_c("'sloughgpt shell'", _CYAN)} then {_c("'confirm on'", _CYAN)} to skip all download prompts")
    _p(f"    {_c('•', _GREEN)} Run {_c("'sloughgpt shell'", _CYAN)} for 40+ built-in commands")
    _p(f"    {_c('•', _GREEN)} Add {_c('--yes/-y', _CYAN)} to skip confirmations for a single command")
    _p()


# ── Category definitions ────────────────────────────────────────────────

_CATEGORIES = {
    "Getting Started": {
        "cmds": ["start", "chat", "shell", "tui"],
        "desc": "New here? Start with these",
    },
    "Server": {
        "cmds": ["dev", "serve", "hf-serve"],
        "desc": "Run inference API",
    },
    "Models": {
        "cmds": ["model", "personality"],
        "desc": "Load, switch, and manage AI models",
    },
    "Training": {
        "cmds": ["train", "checkpoint", "adapter", "feedback"],
        "desc": "Fine-tune and evaluate models",
    },
    "Data": {
        "cmds": ["dataset", "knowledge", "experiment", "collect"],
        "desc": "Import and manage training data",
    },
    "Intelligence": {
        "cmds": ["tokenizer", "vector", "meta-weights", "learn", "memory", "token-tree"],
        "desc": "AI features and tokenization",
    },
    "Media": {
        "cmds": ["images", "multimodal", "companion"],
        "desc": "Images, vision, and AI companion",
    },
    "System": {
        "cmds": ["system", "error", "completion", "simulate", "security", "docstore", "feeds", "logs", "monitor"],
        "desc": "Environment, diagnostics, and storage",
    },
    "Docker": {
        "cmds": ["docker", "build", "vm", "world"],
        "desc": "Containerized deployment and infrastructure",
    },
    "Advanced": {
        "cmds": ["agent", "session", "generate"],
        "desc": "AI agents, chat sessions, and generation",
    },
}


# ── Post-command suggestions ───────────────────────────────────────────

_SUGGESTIONS = {
    "model": "Tip: try `model list`, `model status` next",
    "train": "Tip: try `train monitor`, `train eval` next",
    "dataset": "Tip: try `dataset list`, `dataset stats` next",
    "knowledge": "Tip: try `knowledge search`, `knowledge gaps` next",
    "memory": "Tip: try `memory stats`, `memory search` next",
    "adapter": "Tip: try `adapter list`, `adapter info` next",
    "checkpoint": "Tip: try `checkpoint list`, `checkpoint load` next",
    "system": "Tip: try `system status`, `system health` next",
    "error": "Tip: try `error recent`, `error grouped` next",
    "experiment": "Tip: try `experiment list`, `experiment create` next",
    "personality": "Tip: try `personality list`, `personality load` next",
    "agent": "Tip: try `agent list`, `agent create` next",
    "session": "Tip: try `session list`, `session messages` next",
    "feedback": "Tip: try `feedback export`, `feedback prepare` next",
    "images": "Tip: try `images generate`, `images gallery` next",
    "companion": "Tip: try `companion status`, `companion chat` next",
    "learn": "Tip: try `learn search`, `learn status` next",
    "tokenizer": "Tip: try `tokenizer tokenize`, `tokenizer vocab` next",
    "vector": "Tip: try `vector init`, `vector search` next",
    "collect": "Tip: try `collect file`, `collect url` next",
    "security": "Tip: try `security audit`, `security keys` next",
    "docstore": "Tip: try `docstore collections`, `docstore list` next",
    "feeds": "Tip: try `feeds rss`, `feeds json` next",
    "meta-weights": "Tip: try `meta-weights get`, `meta-weights stats` next",
    "multimodal": "Tip: try `multimodal status`, `multimodal dpo` next",
}


# ── Main entry ──────────────────────────────────────────────────────────

def run(group: Group, args: Optional[List[str]] = None):
    """Run the CLI. This is the main entry point."""
    if args is None:
        args = sys.argv[1:]

    ctx = Context()
    ctx.ensure_object()

    # Parse global options first
    global_opts, remaining = _parse_global_options(args)
    ctx.obj.update(global_opts)

    # Handle --help
    if ctx.obj.get("help"):
        _format_help(group, ctx)
        return

    # Handle --version
    if ctx.obj.get("version"):
        try:
            from core.version import format_version_display
            print(format_version_display())
        except Exception:
            print("sloughgpt v0.3.0")
        return

    # Handle no command
    if not remaining:
        if group.invoke_without_command:
            _format_help(group, ctx)
            return
        _format_help(group, ctx)
        return

    cmd_name = remaining[0]
    cmd_args = remaining[1:]

    # Try to find command or group
    result = _resolve_and_run(group, ctx, cmd_name, cmd_args)
    if result is None:
        return
    cmd_name, cmd_args = result

    # Show post-command suggestion
    if _TTY:
        suggestion = _SUGGESTIONS.get(cmd_name)
        if suggestion:
            _p(f"\n  {_c(suggestion, _DIM)}")


def _parse_global_options(args: List[str]) -> Tuple[dict, List[str]]:
    """Extract global options (--host, --port, --json, etc.) from args."""
    global_opts = {}
    remaining = []
    i = 0

    while i < len(args):
        arg = args[i]

        if arg == "--host" and i + 1 < len(args):
            global_opts["host"] = args[i + 1]
            i += 2
        elif arg == "--port" and i + 1 < len(args):
            global_opts["port"] = int(args[i + 1])
            i += 2
        elif arg in ("-c", "--config") and i + 1 < len(args):
            global_opts["config"] = args[i + 1]
            i += 2
        elif arg == "--json":
            global_opts["json"] = True
            i += 1
        elif arg == "--no-color":
            global_opts["no_color"] = True
            i += 1
        elif arg in ("-q", "--quiet"):
            global_opts["quiet"] = True
            i += 1
        elif arg == "--timeout" and i + 1 < len(args):
            global_opts["timeout"] = int(args[i + 1])
            i += 2
        elif arg == "--version":
            global_opts["version"] = True
            i += 1
        elif arg in ("--help", "-h"):
            global_opts["help"] = True
            i += 1
        elif arg in ("--yes", "-y"):
            global_opts["yes"] = True
            i += 1
        else:
            remaining.append(arg)
            i += 1

    # Set defaults
    global_opts.setdefault("host", "localhost")
    global_opts.setdefault("port", 8000)
    global_opts.setdefault("timeout", 10)

    return global_opts, remaining


def _resolve_and_run(group: Group, ctx: Context, cmd_name: str, cmd_args: List[str]
                     ) -> Optional[Tuple[str, List[str]]]:
    """Resolve a command name and run it. Returns (name, args) on success."""
    # Try exact match in groups
    if cmd_name in group.groups:
        sub = group.groups[cmd_name]
        _record_usage(cmd_name)
        return _run_group(sub, ctx, cmd_args)

    # Try exact match in commands
    if cmd_name in group.commands:
        cmd = group.commands[cmd_name]
        _record_usage(cmd_name)
        _run_command(cmd, ctx, cmd_args)
        return (cmd_name, cmd_args)

    # Fuzzy match
    matches = group._fuzzy_match(cmd_name)
    if matches:
        best = matches[0]

        # Auto-correct prompt
        if _TTY and sys.stdin.isatty():
            _p()
            _p(f"  {_c('?', _YELLOW)} {_c('Unknown command: ', _DIM)}{_c(cmd_name, _RED)}")
            _p(f"  {_c('→', _GREEN)} {_c('Did you mean ', _DIM)}{_c(best, _CYAN + _BOLD)}{_c('?', _DIM)}")
            try:
                answer = input("    [Y/n] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = "n"
            if answer in ("", "y", "yes"):
                return _resolve_and_run(group, ctx, best, cmd_args)
            # Declined — show error
            _show_error(group, cmd_name)
            return None
        else:
            # Non-interactive: auto-resolve silently
            return _resolve_and_run(group, ctx, best, cmd_args)

    # No match — show error
    _show_error(group, cmd_name)
    return None


def _run_group(group: Group, ctx: Context, args: List[str]
               ) -> Optional[Tuple[str, List[str]]]:
    """Run a group's subcommand."""
    if not args:
        if group.invoke_without_command:
            return None
        _format_help(group, ctx)
        return None

    cmd_name = args[0]
    cmd_args = args[1:]

    return _resolve_and_run(group, ctx, cmd_name, cmd_args)


def _run_command(cmd: Command, ctx: Context, args: List[str]):
    """Parse args and invoke a command."""
    try:
        kwargs, extra = _parse_args(args, cmd.options, cmd.arguments)
    except ValueError as e:
        _p(f"  {_c('Error:', _RED)} {e}")
        sys.exit(1)

    # Add context to kwargs
    kwargs["ctx"] = ctx

    try:
        cmd.func(**kwargs)
    except SystemExit:
        raise
    except KeyboardInterrupt:
        _p(f"\n  {_c('Interrupted', _DIM)}")
        sys.exit(130)
    except Exception as e:
        if not ctx.obj.get("quiet"):
            _p(f"  {_c('Error:', _RED)} {e}")
        sys.exit(1)


def _show_error(group: Group, cmd_name: str):
    """Show error with available commands."""
    _p()
    _p(f"  {_c('✗', _RED)} {_c('Unknown command: ', _BOLD)}{_c(cmd_name, _CYAN)}")
    _p()

    all_names = sorted(list(group.commands.keys()) + list(group.groups.keys()))
    if all_names:
        _p(f"  {_c('Available commands:', _BOLD)}")
        for name in all_names:
            _p(f"    {_c(name, _CYAN)}")
        _p()

        _p(f"  {_c('Tip: Use \'sloughgpt --help\' to see all commands', _DIM)}")
        _p()


# ── Decorator helpers (drop-in for click decorators) ────────────────────

def option(*names, help="", default=None, type=str, is_flag=False,
           required=False, multiple=False, metavar="", show_default=False,
           choice=None):
    """Decorator to add an option to a command."""
    opt = Option(list(names), help=help, default=default, type=type,
                 is_flag=is_flag, required=required, multiple=multiple,
                 metavar=metavar, show_default=show_default, choice=choice)
    def decorator(func):
        if not hasattr(func, "_options"):
            func._options = []
        func._options.append(opt)
        return func
    return decorator


def argument(name, required=True, default=None, nargs=1):
    """Decorator to add a positional argument to a command."""
    arg = Argument(name, required=required, default=default, nargs=nargs)
    def decorator(func):
        if not hasattr(func, "_arguments"):
            func._arguments = []
        func._arguments.append(arg)
        return func
    return decorator


def pass_context(func):
    """Decorator to mark that the command needs the context."""
    # We always pass ctx, this is just for compatibility
    return func


# ── click.echo / click.confirm replacements ─────────────────────────────

def echo(message: str = "", nl: bool = True, err: bool = False):
    """Print a message (like click.echo)."""
    stream = sys.stderr if err else sys.stdout
    if message:
        stream.write(str(message))
    if nl:
        stream.write("\n")
    stream.flush()


def confirm(message: str, abort: bool = False) -> bool:
    """Prompt for confirmation (like click.confirm)."""
    try:
        answer = input(f"{message} [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        if abort:
            raise SystemExit(1)
        return False
    if answer not in ("y", "yes"):
        if abort:
            raise SystemExit(1)
        return False
    return True
