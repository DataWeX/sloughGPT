"""
slo_cli — Custom CLI framework for SloughGPT. Replaces Click.

Provides: group, command, option, argument, pass_context, echo, confirm,
Choice, Path, Context with invoke/obj/invoked_subcommand, auto-correct,
fuzzy matching, grouped help, usage tracking.
"""

import os
import sys
import json
from difflib import get_close_matches
from pathlib import Path as _Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


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

_USAGE_PATH = _Path.home() / ".config" / "sloughgpt" / "usage_stats.json"

def _record_usage(cmd_name: str) -> None:
    try:
        _USAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        if _USAGE_PATH.exists():
            data = json.loads(_USAGE_PATH.read_text())
        data[cmd_name] = data.get(cmd_name, 0) + 1
        _USAGE_PATH.write_text(json.dumps(data, indent=2))
    except Exception:
        pass


# ── echo / confirm ──────────────────────────────────────────────────────

def echo(message: str = "", nl: bool = True, err: bool = False):
    stream = sys.stderr if err else sys.stdout
    if message:
        stream.write(str(message))
    if nl:
        stream.write("\n")
    stream.flush()


def confirm(message: str, abort: bool = False) -> bool:
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


# ── Types ───────────────────────────────────────────────────────────────

class Choice:
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
        raise BadParameter(f"Invalid value '{value}' for {param_name}. Choose from: {valid}")


class Path:
    def __init__(self, exists=False, file_okay=True, dir_okay=True,
                 writable=False, readable=True, resolve_path=False):
        self.exists = exists
        self.file_okay = file_okay
        self.dir_okay = dir_okay
        self.writable = writable
        self.readable = readable
        self.resolve_path = resolve_path

    def convert(self, value: str, param_name: str) -> str:
        p = _Path(value)
        if self.resolve_path:
            p = p.resolve()
            value = str(p)
        if self.exists and not p.exists():
            raise BadParameter(f"Path '{value}' does not exist")
        if p.exists():
            if not self.file_okay and p.is_file():
                raise BadParameter(f"Path '{value}' is a file")
            if not self.dir_okay and p.is_dir():
                raise BadParameter(f"Path '{value}' is a directory")
        return value


class IntRange:
    def __init__(self, min=None, max=None):
        self.min = min
        self.max = max

    def convert(self, value: str, param_name: str) -> int:
        try:
            iv = int(value)
        except ValueError:
            raise BadParameter(f"'{value}' is not a valid integer")
        if self.min is not None and iv < self.min:
            raise BadParameter(f"{iv} is less than minimum {self.min}")
        if self.max is not None and iv > self.max:
            raise BadParameter(f"{iv} is greater than maximum {self.max}")
        return iv


# ── Exceptions ──────────────────────────────────────────────────────────

class UsageError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class BadParameter(UsageError):
    pass


# ── Parameter definitions ───────────────────────────────────────────────

class Option:
    def __init__(self, names: List[str], help: str = "", default: Any = None,
                 type: type = str, is_flag: bool = False, required: bool = False,
                 multiple: bool = False, metavar: str = "", show_default: bool = False,
                 choice: Optional[Choice] = None, flag_value: Optional[str] = None):
        # Expand slash-separated names: "--tui/--no-tui" → ["--tui", "--no-tui"]
        expanded = []
        for n in names:
            if "/" in n:
                expanded.extend([x.strip() for x in n.split("/") if x.strip()])
            else:
                expanded.append(n)
        self.names = expanded
        self.help = help
        self.default = default
        self.type = type
        self.is_flag = is_flag
        self.required = required
        self.multiple = multiple
        self.metavar = metavar
        self.show_default = show_default
        self.choice = choice
        self.flag_value = flag_value
        # Handle --no- prefix for bool flags
        self.is_bool_flag = is_flag and any(n.startswith("--no-") for n in self.names)
        # dest is from the first long option (--name → name)
        self.dest = None
        for n in self.names:
            if n.startswith("--"):
                self.dest = n.lstrip("-").replace("-", "_")
                break
        if self.dest is None and self.names:
            self.dest = self.names[0].lstrip("-").replace("-", "_")
        if self.is_bool_flag:
            for n in self.names:
                if not n.startswith("--no-"):
                    self.dest = n.lstrip("-").replace("-", "_")
                    break

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
    def __init__(self, name: str, required: bool = True, default: Any = None,
                 nargs: int = 1, type=None):
        self.name = name
        self.required = required
        self.default = default
        self.nargs = nargs  # -1 = variadic
        self.type = type


# ── Context ─────────────────────────────────────────────────────────────

class Context:
    def __init__(self, obj: Optional[dict] = None):
        self.obj = obj or {}
        self.invoked_subcommand: Optional[str] = None
        self._parent: Optional["Context"] = None
        self._command_name: str = ""

    def ensure_object(self, factory: Callable = dict):
        if not self.obj:
            self.obj = factory()

    def invoke(self, cmd, **kwargs):
        """Invoke another command with given kwargs."""
        if isinstance(cmd, Command):
            cmd.func(**kwargs)
        elif callable(cmd):
            cmd(**kwargs)


# ── Command ─────────────────────────────────────────────────────────────

class Command:
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
    def __init__(self, name: str = "", help: str = "",
                 invoke_without_command: bool = False,
                 cls=None):
        self.name = name
        self.help = help
        self.commands: Dict[str, Command] = {}
        self.groups: Dict[str, "Group"] = {}
        self.parent: Optional["Group"] = None
        self.invoke_without_command = invoke_without_command

    def command(self, name: str = "", help: str = "", hidden: bool = False):
        """Decorator to register a command on this group."""
        def decorator(func):
            cmd_name = name or func.__name__
            cmd = Command(cmd_name, func, help, hidden=hidden)
            cmd.options = getattr(func, "_options", [])
            cmd.arguments = getattr(func, "_arguments", [])
            self.commands[cmd_name] = cmd
            return func
        return decorator

    def group(self, name: str = "", help: str = ""):
        """Decorator to create and register a subgroup."""
        def decorator(func):
            grp_name = name or func.__name__
            grp = Group(grp_name, help)
            grp.parent = self
            # Store the original function as the group callback
            grp.callback = func
            grp._options = getattr(func, "_options", [])
            self.groups[grp_name] = grp
            return grp
        return decorator

    def add_command(self, cmd: Command, name: str = ""):
        self.commands[name or cmd.name] = cmd

    def add_group(self, grp: "Group", name: str = ""):
        self.groups[name or grp.name] = grp

    def _fuzzy_match(self, cmd_name: str) -> List[str]:
        all_names = list(self.commands.keys()) + list(self.groups.keys())
        prefix = [c for c in all_names if c.startswith(cmd_name.lower())]
        if prefix:
            return prefix
        substring = [c for c in all_names if cmd_name.lower() in c.lower()]
        if substring:
            return substring
        cutoff = 0.4 if len(cmd_name) <= 3 else 0.6
        return get_close_matches(cmd_name, all_names, n=3, cutoff=cutoff)

    def __call__(self, *args, **kwargs):
        """Make Group callable — runs the CLI."""
        run(self)


# ── Module-level decorators (click-compatible) ──────────────────────────

def group(name=None, help="", invoke_without_command=False, cls=None, **kwargs):
    """Module-level group decorator. Used as @click.group(...)."""
    def decorator(func):
        grp_name = name or func.__name__
        grp = Group(grp_name, help, invoke_without_command=invoke_without_command)
        grp.callback = func
        grp._options = getattr(func, "_options", [])
        # Copy version_option metadata
        if hasattr(func, "_version_option"):
            grp._version_option = func._version_option
            grp._version_package = getattr(func, "_version_package", "")
            grp._version_prog = getattr(func, "_version_prog", "")
            grp._version_value = getattr(func, "_version_value", None)
        return grp
    return decorator


def command(name=None, help="", hidden=False, cls=None, **kwargs):
    """Module-level command decorator. Used as @click.command(...)."""
    def decorator(func):
        cmd_name = name or func.__name__
        cmd = Command(cmd_name, func, help, hidden=hidden)
        cmd.options = getattr(func, "_options", [])
        cmd.arguments = getattr(func, "_arguments", [])
        return cmd
    return decorator


# ── Decorators ──────────────────────────────────────────────────────────

def option(*names, help="", default=None, type=str, is_flag=False,
           required=False, multiple=False, metavar="", show_default=False,
           choice=None, flag_value=None):
    opt = Option(list(names), help=help, default=default, type=type,
                 is_flag=is_flag, required=required, multiple=multiple,
                 metavar=metavar, show_default=show_default, choice=choice,
                 flag_value=flag_value)
    def decorator(func):
        if not hasattr(func, "_options"):
            func._options = []
        func._options.append(opt)
        return func
    return decorator


def argument(name, required=True, default=None, nargs=1, type=None):
    arg = Argument(name, required=required, default=default, nargs=nargs, type=type)
    def decorator(func):
        if not hasattr(func, "_arguments"):
            func._arguments = []
        func._arguments.append(arg)
        return func
    return decorator


def pass_context(func):
    return func


def version_option(package_name="", prog_name="", version=None, **kwargs):
    """Mark command to show --version."""
    def decorator(func):
        func._version_option = True
        func._version_package = package_name
        func._version_prog = prog_name
        func._version_value = version
        return func
    return decorator


def confirmation_option(**kwargs):
    """Add --yes/-y flag."""
    def decorator(func):
        if not hasattr(func, "_options"):
            func._options = []
        func._options.append(Option(
            ["--yes", "-y"], help="Skip confirmation prompt",
            is_flag=True, default=False
        ))
        return func
    return decorator


def password_option(**kwargs):
    return option("--password", help="Password", **kwargs)


# ── Parser ──────────────────────────────────────────────────────────────

def _parse_args(args: List[str], options: List[Option], arguments: List[Argument]
               ) -> Tuple[dict, List[str]]:
    kwargs = {}
    positional = []
    i = 0

    # Set defaults
    for opt in options:
        if opt.is_flag:
            if opt.is_bool_flag:
                kwargs[opt.dest] = None  # Not set yet
            else:
                kwargs[opt.dest] = False
        elif opt.multiple:
            kwargs[opt.dest] = []
        elif opt.default is not None:
            kwargs[opt.dest] = opt.default

    while i < len(args):
        arg = args[i]

        if arg.startswith("-") and arg != "-":
            matched = False
            for opt in options:
                if arg in opt.names:
                    if opt.is_flag:
                        if opt.is_bool_flag:
                            # --no- prefix means False
                            if arg.startswith("--no-"):
                                kwargs[opt.dest] = False
                            else:
                                kwargs[opt.dest] = True
                        else:
                            kwargs[opt.dest] = True
                        matched = True
                        break
                    else:
                        i += 1
                        if i >= len(args):
                            raise UsageError(f"Option {arg} requires a value")
                        value = args[i]
                        if opt.choice:
                            value = opt.choice.convert(value, opt.primary)
                        elif opt.type == int:
                            value = int(value)
                        elif opt.type == float:
                            value = float(value)
                        elif isinstance(opt.type, (Choice, Path, IntRange)):
                            value = opt.type.convert(value, opt.primary)
                        if opt.multiple:
                            kwargs.setdefault(opt.dest, []).append(value)
                        else:
                            kwargs[opt.dest] = value
                        matched = True
                        break

            if not matched:
                # --key=VALUE
                if "=" in arg:
                    key, value = arg.split("=", 1)
                    for opt in options:
                        if key in opt.names:
                            if opt.choice:
                                value = opt.choice.convert(value, opt.primary)
                            elif opt.type == int:
                                value = int(value)
                            elif opt.type == float:
                                value = float(value)
                            elif isinstance(opt.type, (Choice, Path, IntRange)):
                                value = opt.type.convert(value, opt.primary)
                            kwargs[opt.dest] = value
                            matched = True
                            break
                if not matched:
                    raise UsageError(f"Unknown option: {arg}")
        else:
            positional.append(arg)
        i += 1

    # Validate required options
    for opt in options:
        if opt.required and opt.dest not in kwargs:
            raise UsageError(f"Missing required option: {opt.primary}")

    # Map positional args to arguments
    pos_idx = 0
    for arg_def in arguments:
        if arg_def.nargs == -1:
            kwargs[arg_def.name] = positional[pos_idx:]
            pos_idx = len(positional)
        elif pos_idx < len(positional):
            val = positional[pos_idx]
            if arg_def.type and isinstance(arg_def.type, (Choice, Path, IntRange)):
                val = arg_def.type.convert(val, arg_def.name)
            kwargs[arg_def.name] = val
            pos_idx += 1
        elif arg_def.required:
            raise UsageError(f"Missing required argument: {arg_def.name}")
        else:
            kwargs[arg_def.name] = arg_def.default

    return kwargs, positional[pos_idx:]


# ── Help formatting ─────────────────────────────────────────────────────

def _format_help(group: Group, ctx: Context) -> str:
    lines = []
    try:
        from core.version import format_version_display
        version = format_version_display()
    except Exception:
        version = "dev"

    lines.append(f"\n  {_c('SloughGPT', _BOLD + _CYAN)} {_c(f'({version})', _DIM)}\n")
    lines.append(f"  {_c('Train, chat, serve, and manage AI models', _DIM)}\n")

    _p()
    _p(f"  {_c('Commands:', _BOLD)}")

    printed = set()
    for cat_name, cat_info in _CATEGORIES.items():
        cat_items = []
        for cmd_name in cat_info["cmds"]:
            if cmd_name in group.commands:
                cat_items.append(("cmd", group.commands[cmd_name]))
            elif cmd_name in group.groups:
                cat_items.append(("grp", group.groups[cmd_name]))
        if cat_items:
            desc = cat_info["desc"]
            _p(f"\n    {_c(cat_name, _BOLD + _YELLOW)} {_c(f'— {desc}', _DIM)}")
            for kind, item in cat_items:
                if kind == "cmd" and item.hidden:
                    continue
                padded = item.name.ljust(16)
                help_text = (item.help or "")[:50]
                _p(f"      {_c(padded, _CYAN)} {_c(help_text, _DIM)}")
                printed.add(item.name)

    # Remaining
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
            help_text = (item.help or "")[:50]
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


def _p(text: str = "") -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


# ── Categories ──────────────────────────────────────────────────────────

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


# ── Entry point ─────────────────────────────────────────────────────────

def run(group: Group, args: Optional[List[str]] = None):
    """Main entry point. Parses args, resolves commands, runs them."""
    if args is None:
        args = sys.argv[1:]

    ctx = Context()
    ctx.ensure_object()

    # Parse global options
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
            echo(format_version_display())
        except Exception:
            echo("sloughgpt v0.3.0")
        return

    # Handle no command
    if not remaining:
        if group.invoke_without_command:
            _format_help(group, ctx)
            return
        _format_help(group, ctx)
        return

    # If root group has a callback, call it
    if hasattr(group, 'callback') and group.callback:
        ctx.invoked_subcommand = remaining[0] if remaining else None
        main_kwargs = {}
        for opt in getattr(group, '_options', []):
            if opt.dest in ctx.obj:
                main_kwargs[opt.dest] = ctx.obj[opt.dest]
        import inspect
        sig = inspect.signature(group.callback)
        if "ctx" in sig.parameters:
            main_kwargs['ctx'] = ctx
        try:
            group.callback(**main_kwargs)
        except SystemExit:
            raise
        except Exception:
            pass

    cmd_name = remaining[0]
    cmd_args = remaining[1:]

    result = _resolve_and_run(group, ctx, cmd_name, cmd_args)

    # Post-command suggestion
    if _TTY and result:
        suggestion = _SUGGESTIONS.get(result[0])
        if suggestion:
            _p(f"\n  {_c(suggestion, _DIM)}")


def _parse_global_options(args: List[str]) -> Tuple[dict, List[str]]:
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

    global_opts.setdefault("host", "localhost")
    global_opts.setdefault("port", 8000)
    global_opts.setdefault("timeout", 10)

    return global_opts, remaining


def _resolve_and_run(group: Group, ctx: Context, cmd_name: str, cmd_args: List[str]
                     ) -> Optional[Tuple[str, List[str]]]:
    # Exact match in groups
    if cmd_name in group.groups:
        sub = group.groups[cmd_name]
        _record_usage(cmd_name)
        return _run_group(sub, ctx, cmd_args)

    # Exact match in commands
    if cmd_name in group.commands:
        cmd = group.commands[cmd_name]
        _record_usage(cmd_name)
        _run_command(cmd, ctx, cmd_args)
        return (cmd_name, cmd_args)

    # Fuzzy match
    matches = group._fuzzy_match(cmd_name)
    if matches:
        best = matches[0]
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
            _show_error(group, cmd_name)
            return None
        else:
            return _resolve_and_run(group, ctx, best, cmd_args)

    _show_error(group, cmd_name)
    return None


def _run_group(group: Group, ctx: Context, args: List[str]
               ) -> Optional[Tuple[str, List[str]]]:
    if not args:
        if group.invoke_without_command:
            return None
        _format_help(group, ctx)
        return None

    cmd_name = args[0]
    cmd_args = args[1:]
    return _resolve_and_run(group, ctx, cmd_name, cmd_args)


def _run_command(cmd: Command, ctx: Context, args: List[str]):
    try:
        kwargs, extra = _parse_args(args, cmd.options, cmd.arguments)
    except UsageError as e:
        _p(f"  {_c('Error:', _RED)} {e}")
        sys.exit(1)

    # Add context if function accepts it
    import inspect
    sig = inspect.signature(cmd.func)
    if "ctx" in sig.parameters:
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
