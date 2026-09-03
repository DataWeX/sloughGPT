"""
SloughGPT CLI — pure Python entry point with ANSI output.

Commands organized into logical groups. All delegate to existing
cmd_* functions in commands/ modules.
"""

import logging
import sys
import os
from pathlib import Path
from types import SimpleNamespace
import json
from difflib import get_close_matches
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Ensure both CLI core and core-py domains are on the path
_CLI_DIR = Path(__file__).resolve().parent
_CORE_PY_DIR = _CLI_DIR.parent.parent.parent / "packages" / "core-py"
for _sys_path in [_CLI_DIR, str(_CORE_PY_DIR)]:
    if str(_sys_path) not in sys.path:
        sys.path.insert(0, str(_sys_path))

# ── Structured logging (centralized, CLI uses CLILogger via BridgeHandler)
from domains.logging.config import setup_logging  # noqa: E402
from domains.logging import CLILogger, BridgeHandler, set_global  # noqa: E402

setup_logging(enable_console=False, enable_output_buffer=False)
log = CLILogger("slo")
set_global(log)
_bridge = BridgeHandler(log)
logging.root.addHandler(_bridge)

from core.version import format_version_display  # noqa: E402

# ── Inline CLI framework (replaces Click) ──────────────────────────────

_TTY = sys.stdout.isatty()

def _c(text: str, code: str) -> str:
    return f"{code}{text}\033[0m" if _TTY else text

_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"

_USAGE_PATH = Path.home() / ".config" / "sloughgpt" / "usage_stats.json"

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

class CliPath:
    def __init__(self, exists=False, file_okay=True, dir_okay=True,
                 writable=False, readable=True, resolve_path=False):
        self.exists = exists
        self.file_okay = file_okay
        self.dir_okay = dir_okay
        self.writable = writable
        self.readable = readable
        self.resolve_path = resolve_path
    def convert(self, value: str, param_name: str) -> str:
        p = Path(value)
        if self.resolve_path:
            p = p.resolve()
            value = str(p)
        if self.exists and not p.exists():
            raise BadParameter(f"Path '{value}' does not exist")
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

class UsageError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)

class BadParameter(UsageError):
    pass

class Option:
    def __init__(self, names: List[str], help: str = "", default: Any = None,
                 type: type = str, is_flag: bool = False, required: bool = False,
                 multiple: bool = False, metavar: str = "", show_default: bool = False,
                 choice: Optional[Choice] = None, flag_value: Optional[str] = None):
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
        self.is_bool_flag = is_flag and any(n.startswith("--no-") for n in self.names)
        self.dest = None
        for n in self.names:
            if not n.startswith("-"):
                self.dest = n
                break
        if self.dest is None:
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
        self.nargs = nargs
        self.type = type

class Context:
    def __init__(self, obj: Optional[dict] = None):
        self.obj = obj or {}
        self.invoked_subcommand: Optional[str] = None
    def ensure_object(self, factory: Callable = dict):
        if not self.obj:
            self.obj = factory()
    def invoke(self, cmd, **kwargs):
        if isinstance(cmd, Command):
            cmd.func(**kwargs)
        elif callable(cmd):
            cmd(**kwargs)

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

class Group:
    def __init__(self, name: str = "", help: str = "",
                 invoke_without_command: bool = False, cls=None):
        self.name = name
        self.help = help
        self.commands: Dict[str, Command] = {}
        self.groups: Dict[str, "Group"] = {}
        self.parent: Optional["Group"] = None
        self.invoke_without_command = invoke_without_command
    def command(self, name: str = "", help: str = "", hidden: bool = False):
        def decorator(func):
            cmd_name = name or func.__name__
            cmd = Command(cmd_name, func, help, hidden=hidden)
            cmd.options = getattr(func, "_options", [])
            cmd.arguments = getattr(func, "_arguments", [])
            self.commands[cmd_name] = cmd
            return func
        return decorator
    def group(self, name: str = "", help: str = ""):
        def decorator(func):
            grp_name = name or func.__name__
            grp = Group(grp_name, help)
            grp.parent = self
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
        run(self)

def group(name=None, help="", invoke_without_command=False, cls=None, **kwargs):
    def decorator(func):
        grp_name = name or func.__name__
        grp = Group(grp_name, help, invoke_without_command=invoke_without_command)
        grp.callback = func
        grp._options = getattr(func, "_options", [])
        if hasattr(func, "_version_option"):
            grp._version_option = func._version_option
            grp._version_package = getattr(func, "_version_package", "")
            grp._version_prog = getattr(func, "_version_prog", "")
            grp._version_value = getattr(func, "_version_value", None)
        return grp
    return decorator

def command(name=None, help="", hidden=False, cls=None, **kwargs):
    def decorator(func):
        cmd_name = name or func.__name__
        cmd = Command(cmd_name, func, help, hidden=hidden)
        cmd.options = getattr(func, "_options", [])
        cmd.arguments = getattr(func, "_arguments", [])
        return cmd
    return decorator

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
    def decorator(func):
        func._version_option = True
        func._version_package = package_name
        func._version_prog = prog_name
        func._version_value = version
        return func
    return decorator

def confirmation_option(**kwargs):
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

def _parse_args(args: List[str], options: List[Option], arguments: List[Argument]
               ) -> Tuple[dict, List[str]]:
    kwargs = {}
    positional = []
    i = 0
    for opt in options:
        if opt.is_flag:
            if opt.is_bool_flag:
                kwargs[opt.dest] = None
            else:
                kwargs[opt.dest] = False
        elif opt.multiple:
            kwargs[opt.dest] = []
        elif opt.default is not None:
            kwargs[opt.dest] = opt.default
        else:
            kwargs[opt.dest] = None
    while i < len(args):
        arg = args[i]
        if arg.startswith("-") and arg != "-":
            matched = False
            for opt in options:
                if arg in opt.names:
                    if opt.is_flag:
                        if opt.is_bool_flag:
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
                        elif isinstance(opt.type, (Choice, CliPath, IntRange)):
                            value = opt.type.convert(value, opt.primary)
                        if opt.multiple:
                            kwargs.setdefault(opt.dest, []).append(value)
                        else:
                            kwargs[opt.dest] = value
                        matched = True
                        break
            if not matched:
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
                            elif isinstance(opt.type, (Choice, CliPath, IntRange)):
                                value = opt.type.convert(value, opt.primary)
                            kwargs[opt.dest] = value
                            matched = True
                            break
                if not matched:
                    raise UsageError(f"Unknown option: {arg}")
        else:
            # First non-option arg: subcommand name or positional.
            # Stop here — everything from this point on belongs to the
            # subcommand and is forwarded untouched.
            positional.extend(args[i:])
            break
        i += 1
    for opt in options:
        if opt.required and opt.dest not in kwargs:
            raise UsageError(f"Missing required option: {opt.primary}")
    pos_idx = 0
    for arg_def in arguments:
        if arg_def.nargs == -1:
            kwargs[arg_def.name] = positional[pos_idx:]
            pos_idx = len(positional)
        elif pos_idx < len(positional):
            val = positional[pos_idx]
            if arg_def.type and isinstance(arg_def.type, (Choice, CliPath, IntRange)):
                val = arg_def.type.convert(val, arg_def.name)
            kwargs[arg_def.name] = val
            pos_idx += 1
        elif arg_def.required:
            raise UsageError(f"Missing required argument: {arg_def.name}")
        else:
            kwargs[arg_def.name] = arg_def.default
    return kwargs, positional[pos_idx:]

def _p(text: str = "") -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()

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

_CATEGORIES = {
    "Getting Started": {"cmds": ["start", "chat", "shell", "tui"], "desc": "New here? Start with these"},
    "Server": {"cmds": ["dev", "serve", "hf-serve"], "desc": "Run inference API"},
    "Models": {"cmds": ["model", "personality"], "desc": "Load, switch, and manage AI models"},
    "Training": {"cmds": ["train", "checkpoint", "adapter", "feedback"], "desc": "Fine-tune and evaluate models"},
    "Data": {"cmds": ["dataset", "knowledge", "experiment", "collect"], "desc": "Import and manage training data"},
    "Intelligence": {"cmds": ["tokenizer", "vector", "meta-weights", "learn", "memory", "token-tree"], "desc": "AI features and tokenization"},
    "Media": {"cmds": ["images", "multimodal", "companion"], "desc": "Images, vision, and AI companion"},
    "System": {"cmds": ["system", "error", "completion", "simulate", "security", "docstore", "feeds", "logs", "monitor"], "desc": "Environment, diagnostics, and storage"},
    "Docker": {"cmds": ["docker", "build", "vm", "world"], "desc": "Containerized deployment and infrastructure"},
    "Advanced": {"cmds": ["agent", "session", "generate"], "desc": "AI agents, chat sessions, and generation"},
}

def _format_help(group: Group, ctx: Context) -> None:
    try:
        version = format_version_display()
    except Exception:
        version = "dev"
    _p()
    _p(f"  {_c('SloughGPT', _BOLD + _CYAN)} {_c(f'({version})', _DIM)}")
    _p(f"  {_c('Train, chat, serve, and manage AI models', _DIM)}")
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
                help_text = item.help or ""
                _p(f"      {_c(padded, _CYAN)} {_c(help_text, _DIM)}")
                printed.add(item.name)
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
            help_text = item.help or ""
            _p(f"      {_c(padded, _CYAN)} {_c(help_text, _DIM)}")
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

def _format_group_help(group: Group, ctx: Context) -> None:
    _p()
    _p(f"  {_c('Usage:', _BOLD)} {group.name} [OPTIONS] COMMAND [ARGS]...")
    if group.help:
        _p(f"\n  {group.help}")
    _p()
    if group.commands:
        _p(f"  {_c('Commands:', _BOLD)}")
        for name in sorted(group.commands.keys()):
            cmd = group.commands[name]
            if cmd.hidden:
                continue
            padded = name.ljust(16)
            help_text = cmd.help or ""
            _p(f"    {_c(padded, _CYAN)} {_c(help_text, _DIM)}")
        _p()
    _p(f"  {_c('--help', _CYAN)}   Show this help message")
    _p()

def _format_command_help(cmd: Command, ctx: Context, cmd_path: str = "") -> None:
    """Show help for a single command."""
    _p()
    name = cmd_path or cmd.name
    _p(f"  {_c('Usage:', _BOLD)} sloughgpt {name} [OPTIONS]")
    if cmd.help:
        _p()
        _p(f"  {cmd.help}")
    if cmd.options:
        _p()
        _p(f"  {_c('Options:', _BOLD)}")
        for opt in cmd.options:
            names = ", ".join(opt.names)
            default_str = ""
            if opt.default is not None and opt.default != "" and not opt.is_flag:
                default_str = f" {_c(f'(default: {opt.default})', _DIM)}"
            required_str = f" {_c('(required)', _RED)}" if opt.required else ""
            _p(f"    {names:<20} {opt.help}{default_str}{required_str}")
    _p()

def _run_command(cmd: Command, ctx: Context, args: List[str], cmd_path: str = "") -> None:
    # Handle --help / -h at command level
    if "--help" in args or "-h" in args:
        _format_command_help(cmd, ctx, cmd_path)
        return
    try:
        kwargs, extra = _parse_args(args, cmd.options, cmd.arguments)
    except UsageError as e:
        _p(f"  {_c('Error:', _RED)} {e}")
        sys.exit(1)
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
    except UsageError as e:
        _p(f"  {_c('Error:', _RED)} {e}")
        sys.exit(1)
    except Exception as e:
        if not ctx.obj.get("quiet"):
            _p(f"  {_c('Error:', _RED)} {e}")
        sys.exit(1)

def _show_error(group: Group, cmd_name: str) -> None:
    _p()
    _p(f"  {_c('err', _RED)} {_c('Unknown command: ', _BOLD)}{_c(cmd_name, _CYAN)}")
    _p()
    all_names = sorted(list(group.commands.keys()) + list(group.groups.keys()))
    if all_names:
        _p(f"  {_c('Available commands:', _BOLD)}")
        for name in all_names:
            _p(f"    {_c(name, _CYAN)}")
        _p()
        _p(f"  {_c('Tip: Use \'sloughgpt --help\' to see all commands', _DIM)}")
        _p()

def _resolve_and_run(group: Group, ctx: Context, cmd_name: str, cmd_args: List[str],
                     full_path: str = "") -> Optional[Tuple[str, List[str]]]:
    if cmd_name in group.groups:
        sub = group.groups[cmd_name]
        _record_usage(cmd_name)
        return _run_group(sub, ctx, cmd_args, full_path=f"{full_path} {cmd_name}" if full_path else cmd_name)
    if cmd_name in group.commands:
        cmd = group.commands[cmd_name]
        _record_usage(cmd_name)
        full = f"{full_path} {cmd_name}" if full_path else cmd_name
        _run_command(cmd, ctx, cmd_args, cmd_path=full)
        return (full, cmd_args)
    matches = group._fuzzy_match(cmd_name)
    if matches:
        best = matches[0]
        if _TTY and sys.stdin.isatty():
            _p()
            _p(f"  {_c('?', _YELLOW)} {_c('Unknown command: ', _DIM)}{_c(cmd_name, _RED)}")
            _p(f"  {_c('>', _GREEN)} {_c('Did you mean ', _DIM)}{_c(best, _CYAN + _BOLD)}{_c('?', _DIM)}")
            try:
                answer = input("    [Y/n] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = "n"
            if answer in ("", "y", "yes"):
                return _resolve_and_run(group, ctx, best, cmd_args, full_path)
            _show_error(group, cmd_name)
            return None
        else:
            return _resolve_and_run(group, ctx, best, cmd_args, full_path)
    _show_error(group, cmd_name)
    return None

def _run_group(group: Group, ctx: Context, args: List[str],
               full_path: str = "") -> Optional[Tuple[str, List[str]]]:
    # Always invoke group callback for invoke_without_command groups
    # (e.g. the top-level cli sets ctx.obj["host"], ctx.obj["port"])
    if group.invoke_without_command:
        cb = group.callback if hasattr(group, 'callback') else None
        if cb:
            import inspect
            sig = inspect.signature(cb)
            grp_opts = getattr(group, "_options", [])
            try:
                kwargs, positional = _parse_args(args, grp_opts, [])
            except UsageError as e:
                _p(f"  {_c('Error:', _RED)} {e}")
                sys.exit(1)
            if "ctx" in sig.parameters:
                kwargs["ctx"] = ctx
            cb(**kwargs)
            args = positional  # forward only subcommand args
    if not args:
        if not group.invoke_without_command:
            _format_group_help(group, ctx)
        return None
    cmd_name = args[0]
    cmd_args = args[1:]
    path = f"{full_path} {cmd_name}" if full_path else cmd_name
    return _resolve_and_run(group, ctx, cmd_name, cmd_args, full_path=path)

def run(group: Group, args: Optional[List[str]] = None):
    """Single entry-point for CLI dispatch.

    Parsing model
    ─────────────
    • ``grp._options`` is the **single source of truth** for every option
      the top-level group accepts (``--host``, ``--port``, ``--config``,
      …).  They are defined once via ``@click.option`` decorators on the
      ``cli`` function and collected into ``grp._options`` by the
      ``group()`` decorator.
    • ``_parse_args`` walks the argv list left-to-right.  The first token
      that is *not* a recognised option (or an option-value) is treated as
      a positional argument and appended to the ``positional`` return
      list.  Parsing stops there — everything from that point on is
      returned untouched as ``remaining_args``.
    • ``--help`` / ``-h`` and ``--version`` are **meta-options**: they are
      recognised by the group's option list but handled *before* the
      group callback runs, so the callback never sees them.
    """
    if args is None:
        args = sys.argv[1:]

    ctx = Context()
    ctx.ensure_object()

    # ── 1. Parse group-level options in one pass ────────────────────────
    #    ``grp._options`` is set by the ``@click.option`` decorators on
    #    the ``cli`` function (see the ``group()`` decorator).
    #    ``_parse_args`` walks argv left-to-right.  The first token that
    #    is *not* a recognised option (or option-value) is treated as a
    #    positional arg and parsing stops there — everything from that
    #    point on is returned as ``positional`` (subcommand + its args).
    grp_opts = list(getattr(group, "_options", [])) + [
        Option(["--help", "-h"],  help="Show this message and exit", is_flag=True),
        Option(["--version"],     help="Show version and exit",      is_flag=True),
    ]
    try:
        kwargs, positional = _parse_args(args, grp_opts, [])
    except UsageError as e:
        _p(f"  {_c('Error:', _RED)} {e}")
        sys.exit(1)

    # Separate meta-options from options that go into ctx / callback.
    show_help = kwargs.pop("help", False)
    show_version = kwargs.pop("version", False)

    # The first positional arg is the subcommand name (or a fuzzy match
    # that _resolve_and_run will resolve).  Everything after it is the
    # subcommand's own argv.
    cmd_name = positional[0] if positional else None
    cmd_args = positional[1:] if len(positional) > 1 else []

    # ── 2. Populate ctx.obj ─────────────────────────────────────────────
    ctx.obj.update(kwargs)
    if cmd_name:
        ctx.invoked_subcommand = cmd_name

    # ── 3. Invoke the group callback (populates ctx.obj with defaults) ──
    if hasattr(group, "callback") and group.callback:
        import inspect
        sig = inspect.signature(group.callback)
        cb_kwargs = dict(kwargs)
        if "ctx" in sig.parameters:
            cb_kwargs["ctx"] = ctx
        group.callback(**cb_kwargs)

    # ── 4. Handle meta-options ──────────────────────────────────────────
    if show_help:
        _format_help(group, ctx)
        return
    if show_version:
        try:
            echo(format_version_display())
        except Exception:
            echo("sloughgpt v0.1.0")
        return

    # ── 5. No subcommand → show help ────────────────────────────────────
    if cmd_name is None:
        _format_help(group, ctx)
        return

    # ── 6. Dispatch to subcommand ───────────────────────────────────────
    result = _resolve_and_run(group, ctx, cmd_name, cmd_args)
    if _TTY and result:
        suggestion = _SUGGESTIONS.get(result[0])
        if suggestion:
            _p(f"\n  {_c(suggestion, _DIM)}")

# ── click namespace (so @click.group, @click.option etc. work) ─────────
from types import SimpleNamespace as _NS
click = _NS(
    group=group, command=command, option=option, argument=argument,
    pass_context=pass_context, version_option=version_option,
    confirmation_option=confirmation_option, echo=echo, confirm=confirm,
    Choice=Choice, Path=CliPath, UsageError=UsageError, BadParameter=BadParameter,
    run=run,
)

# ── End inline framework ────────────────────────────────────────────────


from utils.helpers import chat_repository_root as _chat_repository_root


def _ns(**kwargs) -> SimpleNamespace:
    """Build a SimpleNamespace from keyword arguments."""
    return SimpleNamespace(**kwargs)


def _output(ctx, data, *, plain=None):
    """Unified output helper: --json prints data dict, otherwise prints plain text.

    Usage in commands:
        _output(ctx, {"models": [...]}, plain=f"Found {len(models)} models")
    """
    import json as _json

    if ctx.obj.get("json"):
        echo(_json.dumps(data, indent=2, default=str))
    elif plain is not None:
        echo(plain)
    else:
        for k, v in data.items():
            echo(f"{k}: {v}")


def _confirm(ctx, message, *, force=False):
    """Prompt for confirmation unless --quiet or force=True."""
    if force or ctx.obj.get("quiet"):
        return True
    return confirm(message)


def _verbose(ctx, *args):
    """Print only if not --quiet."""
    if not ctx.obj.get("quiet"):
        echo(" ".join(str(a) for a in args))


# ── Docker helpers ────────────────────────────────────────────────────


def _docker_compose_file():
    return _chat_repository_root() / "infra" / "docker" / "docker-compose.yml"


def _docker_action(action: str, a):
    import subprocess
    compose = _docker_compose_file()
    if not compose.is_file():
        log.error(f"Compose file not found: {compose}")
        return

    if action == "start":
        profile = []
        if getattr(a, "dev", False):
            profile = ["--profile", "dev"]
        elif getattr(a, "gpu", False):
            profile = ["--profile", "gpu"]
        log.step("Starting Docker services...")
        subprocess.run(["docker", "compose", "-f", str(compose), "up", "-d", *profile])
        log.success("Services started")
        subprocess.run(["docker", "compose", "-f", str(compose), "ps"])

    elif action == "stop":
        log.step("Stopping Docker services...")
        subprocess.run(["docker", "compose", "-f", str(compose), "down"])
        log.success("Services stopped")

    elif action == "status":
        subprocess.run(["docker", "compose", "-f", str(compose), "ps"])

    elif action == "logs":
        cmd = ["docker", "compose", "-f", str(compose), "logs", "-f"]
        if getattr(a, "service", None):
            cmd.append(a.service)
        subprocess.run(cmd)

    elif action == "build":
        cmd = ["docker", "compose", "-f", str(compose), "build"]
        if getattr(a, "no_cache", False):
            cmd.append("--no-cache")
        log.step("Building Docker images...")
        subprocess.run(cmd)
        log.success("Build complete")

    elif action == "shell":
        service = getattr(a, "service", "api")
        subprocess.run(["docker", "compose", "-f", str(compose), "exec", service, "/bin/bash"])


# ── Top-level CLI ─────────────────────────────────────────────────────


@click.group(invoke_without_command=True)
@click.version_option(package_name="sloughgpt", prog_name="sloughgpt")
@click.option("--host", default="localhost", help="API hostname", show_default=True)
@click.option("--port", default=8000, type=int, help="API port", show_default=True)
@click.option("-c", "--config", default="config.yaml", help="Config path", show_default=True)
@click.option("--json", "output_json", is_flag=True, help="JSON output for commands")
@click.option("--no-color", is_flag=True, help="Disable ANSI color output")
@click.option("--quiet", "-q", is_flag=True, help="Suppress non-essential output")
@click.option("--timeout", default=10, type=int, help="HTTP timeout in seconds", show_default=True)
@click.pass_context
def cli(ctx, host: str, port: int, config: str, output_json: bool, no_color: bool, quiet: bool, timeout: int):
    """SloughGPT CLI — train, chat, serve, and manage models."""
    ctx.ensure_object(dict)
    ctx.obj["host"] = host
    ctx.obj["port"] = port
    ctx.obj["config"] = config
    ctx.obj["json"] = output_json
    ctx.obj["no_color"] = no_color
    ctx.obj["quiet"] = quiet
    ctx.obj["timeout"] = timeout

    if no_color:
        os.environ["NO_COLOR"] = "1"
        os.environ["SLO_NO_COLOR"] = "1"

    if ctx.invoked_subcommand is None:
        _show_welcome_banner()


def _show_welcome_banner():
    """Show a polished welcome banner with version and quick start."""
    import sys

    # Get version
    try:
        version = format_version_display()
    except Exception:
        version = "dev"

    # ANSI helpers
    def _c(text, code):
        if sys.stdout.isatty():
            return f"{code}{text}\033[0m"
        return text

    _BOLD = "\033[1m"
    _DIM = "\033[2m"
    _CYAN = "\033[36m"
    _GREEN = "\033[32m"
    _YELLOW = "\033[33m"
    _RED = "\033[31m"
    _MAGENTA = "\033[35m"

    _write = sys.stdout.write
    _flush = sys.stdout.flush

    def _line(text=""):
        _write(text + "\n")
        _flush()

    # ── ASCII art header ──────────────────────────────────
    _line()
    _line(f"  {_c('  ┌──────────────────────────────────────┐', _DIM)}")
    _line(f"  {_c('  │', _DIM)}{_c('                                      ', _MAGENTA + _BOLD)}{_c('│', _DIM)}")
    _line(f"  {_c('  │', _DIM)}{_c('   ████████╗██╗     ██████╗            ', _MAGENTA + _BOLD)}{_c('│', _DIM)}")
    _line(f"  {_c('  │', _DIM)}{_c('   ╚══██╔══╝██║     ██╔═══██╗           ', _MAGENTA + _BOLD)}{_c('│', _DIM)}")
    _line(f"  {_c('  │', _DIM)}{_c('      ██║   ██║     ██║   ██║           ', _MAGENTA + _BOLD)}{_c('│', _DIM)}")
    _line(f"  {_c('  │', _DIM)}{_c('      ██║   ██║     ██║   ██║           ', _MAGENTA + _BOLD)}{_c('│', _DIM)}")
    _line(f"  {_c('  │', _DIM)}{_c('      ██║   ███████╗╚██████╔╝           ', _MAGENTA + _BOLD)}{_c('│', _DIM)}")
    _line(f"  {_c('  │', _DIM)}{_c('      ╚═╝   ╚══════╝ ╚═════╝            ', _MAGENTA + _BOLD)}{_c('│', _DIM)}")
    _line(f"  {_c('  │', _DIM)}{_c('                                      ', _MAGENTA + _BOLD)}{_c('│', _DIM)}")
    _line(f"  {_c('  └──────────────────────────────────────┘', _DIM)}")
    _line()
    _line(f"  {_c('  sloughGPT', _BOLD + _CYAN)}  {_c(version, _DIM)}")
    _line(f"  {_c('  ─────────────────────────────────────────', _DIM)}")
    _line()

    # Quick start commands
    _line(f"  {_c('Quick Start:', _BOLD)}")
    _line(f"    {_c('sloughgpt start', _CYAN)}        Getting started guide")
    _line(f"    {_c('sloughgpt chat', _CYAN)}         Start chatting with AI")
    _line(f"    {_c('sloughgpt model list', _CYAN)}   List available models")
    _line(f"    {_c('sloughgpt shell', _CYAN)}        Interactive shell")
    _line()

    # Server status
    _show_server_status()

    _line()
    _line(f"  {_c('Run \'sloughgpt --help\' to see all commands', _DIM)}")
    _line()


def _show_server_status():
    """Check and display server status."""
    import sys
    import requests

    def _c(text, code):
        if sys.stdout.isatty():
            return f"{code}{text}\033[0m"
        return text

    _BOLD = "\033[1m"
    _DIM = "\033[2m"
    _GREEN = "\033[32m"
    _YELLOW = "\033[33m"
    _RED = "\033[31m"

    _write = sys.stdout.write
    _flush = sys.stdout.flush

    def _line(text=""):
        _write(text + "\n")
        _flush()

    _line(f"  {_c('Server Status:', _BOLD)}")

    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        if response.status_code == 200:
            raw = response.json()
            data = raw.get("data", raw)

            if data.get("model_loaded"):
                model = data.get("model_type", "unknown")
                _line(f"    {_c('ok', _GREEN)} Server running (model: {model})")
            else:
                _line(f"    {_c('warn', _YELLOW)} Server running (no model loaded)")
        else:
            _line(f"    {_c('err', _RED)} Server unreachable")
    except requests.exceptions.ConnectionError:
        _line(f"    {_c('·', _DIM)} Server not running")
    except Exception:
        _line(f"    {_c('·', _DIM)} Server status unknown")


# ═══════════════════════════════════════════════════════════════════════
# Welcome & Shell
# ═══════════════════════════════════════════════════════════════════════


@cli.command(help="Welcome guide with next steps")
def start():
    from commands import dev
    root = _chat_repository_root()
    echo(f"""
SloughGPT — getting started
===========================

  1. Install Python package:
       python3 -m pip install -e ".[dev]"

  2. Verify environment:
       sloughgpt system doctor

  3. First training run:
       sloughgpt train quick

  4. HTTP API:
       sloughgpt dev

  5. Terminal UI:
       sloughgpt tui

  6. Web UI (separate terminal):
       cd apps/web && npm install && npm run dev

  7. Colab: sloughgpt_colab.ipynb

Repository: {root}

Version: {format_version_display()}
""")


from commands.logs import logs as _logs_cmd
from commands.monitor import monitor as _monitor_cmd

# Wrap Click commands so our framework can dispatch to them
class _ClickCommandWrapper:
    """Wraps a Click command to work with our inline framework."""
    def __init__(self, click_cmd):
        self.click_cmd = click_cmd
        self.name = click_cmd.name
        self.help = click_cmd.help or ""
        self.options = []
        self.arguments = []
        self.hidden = False
        # Extract params from Click command for display
        for param in click_cmd.params:
            if hasattr(param, 'opts'):
                names = param.opts
                self.options.append(Option(
                    names,
                    help=param.help or "",
                    default=param.default,
                    is_flag=param.is_flag if hasattr(param, 'is_flag') else False,
                    type=type(param.type).__name__ if hasattr(param.type, '__name__') else str,
                ))

    def __call__(self, **kwargs):
        # Build args list from kwargs
        args = []
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, bool):
                if v:
                    args.append(f"--{k.replace('_', '-')}")
            else:
                args.append(f"--{k.replace('_', '-')}={v}")
        self.click_cmd.main(args=args, standalone_mode=False)

cli.add_command(_ClickCommandWrapper(_logs_cmd), 'logs')
cli.add_command(_ClickCommandWrapper(_monitor_cmd), 'monitor')


@cli.command(help="Launch interactive terminal UI (split-pane curses)")
@click.pass_context
def tui(ctx):
    """Launch the split-pane curses TUI."""
    ctx.invoke(shell, command=None, tui=True)


@cli.command(help="Launch interactive shell REPL")
@click.option("--command", "-c", help="Run a single command and exit")
@click.option("--tui/--no-tui", default=None, help="Curses TUI mode (default when TTY)")
@click.option("--line", is_flag=True, help="Force line-mode REPL (no TUI)")
@click.pass_context
def shell(ctx, command, tui, line):
    """Launch the SloughGPT interactive shell REPL."""
    from utils.helpers import ensure_server
    actual_url, _server_proc = ensure_server(host=ctx.obj["host"], port=ctx.obj["port"])
    from domains.shell.repl import ShellREPL
    from domains.shell import DaitRuntime

    os = DaitRuntime(api_url=actual_url)
    # Default to TUI when TTY, line mode when piped or --line
    use_tui = True
    if line:
        use_tui = False
    elif tui is not None:
        use_tui = tui
    elif not sys.stdout.isatty():
        use_tui = False
    repl = ShellREPL(os, use_tui=True if use_tui else None)
    if command:
        commands, is_bg, should_time = repl._parse_pipeline(command)
        if is_bg:
            repl._execute_background(command.rstrip("& ").strip())
        elif len(commands) > 1:
            repl._execute_pipeline(commands, should_time=should_time)
        else:
            expanded = repl._expand_alias(command)
            out = repl._execute_single(expanded, "")
            if out:
                echo(out, nl=False)
    else:
        repl.run()


@cli.command(help="Generate shell completion script")
@click.argument("shell", type=click.Choice(["bash", "zsh", "fish"]), default="bash")
def completion(shell):
    """Print a shell completion script. Source it to enable tab-completion.

    \b
    Examples:
      eval "$(sloughgpt completion bash)"   # bash
      eval "$(sloughgpt completion zsh)"    # zsh
      sloughgpt completion fish | source    # fish
    """
    _shell = shell.lower()
    if _shell == "bash":
        echo(f'eval "$(_{{COMPLETE}}={_shell}_complete {{prog}})"'.replace("{{COMPLETE}}", "_COMPLETE").replace("{{prog}}", "sloughgpt"))
    elif _shell == "zsh":
        echo(f'eval "$(_{{COMPLETE}}={_shell}_complete {{prog}})"'.replace("{{COMPLETE}}", "_COMPLETE").replace("{{prog}}", "sloughgpt"))
    elif _shell == "fish":
        echo(f"source (_{{COMPLETE}}={_shell}_complete sloughgpt | psub)")


# ═══════════════════════════════════════════════════════════════════════
# Serve & Chat
# ═══════════════════════════════════════════════════════════════════════


@cli.command(help="Interactive chat")
@click.option("--no-serve", is_flag=True, help="Don't auto-start server")
@click.pass_context
def chat(ctx, no_serve):
    from commands.chat import cmd_chat
    args = _ns(
        no_serve=no_serve, auto_model=None,
        load_mode="local", device="auto", max_tokens=64,
        temperature=0.7, host=ctx.obj["host"], port=ctx.obj["port"],
    )
    cmd_chat(args)


@cli.command(help="One-shot text generation")
@click.argument("prompt")
@click.option("--model", metavar="NAME_OR_PATH", help="Model override")
@click.option("--max-tokens", default=100, type=int, help="Max tokens", show_default=True)
@click.option("--temperature", default=0.8, type=float, help="Temperature", show_default=True)
@click.pass_context
def generate(ctx, prompt, model, max_tokens, temperature):
    from commands.chat import cmd_generate
    args = _ns(
        prompt=prompt, model=model, max_tokens=max_tokens,
        temperature=temperature, host=ctx.obj["host"], port=ctx.obj["port"],
    )
    cmd_generate(args)


@cli.command(help="Start API + Web dev servers")
@click.option("--model", default=None, help="Model path")
@click.option("--web-port", default=3000, type=int, help="Web dev server port")
@click.option("--watch-web", is_flag=True, help="Watch web files for changes")
@click.option("--auto-download", is_flag=True, help="Skip download confirmation on startup")
@click.pass_context
def dev(ctx, model, web_port, watch_web, auto_download):
    from commands.dev import cmd_dev
    args = _ns(
        model=model, web_port=web_port, watch_web=watch_web,
        port=ctx.obj["port"], host=ctx.obj["host"], auto_download=auto_download,
    )
    cmd_dev(args)


@cli.command(help="Start HTTP inference server (with --web: full FastAPI + frontend, --mobile: API + React Native)")
@click.option("--host", default="localhost", help="Bind address", show_default=True)
@click.option("--port", default=8000, type=int, help="API port", show_default=True)
@click.option("--model", metavar="PATH", help="Model to preload")
@click.option("--web", is_flag=True, help="Start full FastAPI server + Next.js web UI and opens browser")
@click.option("--web-port", default=3000, type=int, help="Web UI port", show_default=True)
@click.option("--mobile", is_flag=True, help="Start FastAPI server + React Native metro bundler")
@click.option("--auto-download", is_flag=True, help="Skip download confirmation on startup")
def serve(host, port, model, web, mobile, web_port, auto_download):
    from commands.dev import cmd_serve
    args = _ns(host=host, port=port, model=model, web=web, mobile=mobile, web_port=web_port, auto_download=auto_download)
    cmd_serve(args)


@cli.command("hf-serve", hidden=True, help="Serve a HuggingFace model via API")
@click.argument("model_name")
@click.option("--mode", type=click.Choice(["api", "local"]), default="local")
@click.option("--device", default="auto")
@click.pass_context
def hf_serve(ctx, model_name, mode, device):
    from commands.dev import cmd_hf_serve
    args = _ns(
        model=model_name, mode=mode, device=device,
        host=ctx.obj["host"], port=ctx.obj["port"],
    )
    cmd_hf_serve(args)


# ═══════════════════════════════════════════════════════════════════════
# model  — list, info, download, export, benchmark, compare
# ═══════════════════════════════════════════════════════════════════════


@cli.group(help="List, inspect, download, export, and benchmark models")
@click.pass_context
def model(ctx):
    pass


@model.command("list", help="List available models")
@click.pass_context
def model_list(ctx):
    from commands.models import cmd_models
    cmd_models(_ns(json_output=ctx.obj.get("json")))


@model.command("status", help="Show cached/downloaded models with sizes")
@click.pass_context
def model_status(ctx):
    from commands.models import _cmd_models_status
    _cmd_models_status(_ns(json_output=ctx.obj.get("json")))


@model.command("info", help="Show checkpoint info")
@click.argument("checkpoint", default="models/sloughgpt.soul")
@click.pass_context
def model_info(ctx, checkpoint):
    from commands.models import _cmd_models_info
    _cmd_models_info(_ns(model=checkpoint, json_output=ctx.obj.get("json")))


@model.command("download", help="Download model from HuggingFace")
@click.argument("model_id", required=False, default=None)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def model_download(ctx, model_id, yes):
    from commands.models import _cmd_models_download
    _cmd_models_download(_ns(model_id=model_id, yes=yes))


@model.command("export", help="Export model to different formats")
@click.argument("checkpoint", default="models/sloughgpt.soul")
@click.option("--output", "-o", help="Output path")
@click.option("--format", "-f", "fmt",
    type=click.Choice(["safetensors", "safetensors_bf16", "onnx", "gguf_q4_k_m",
                       "gguf_fp16", "gguf_q5_k_m", "gguf_q8_0",
                       "sou", "all"]),
    default="safetensors", help="Export format")
@click.option("--quantize", type=click.Choice(["Q4_K_M", "Q5_K_M", "Q8_0", "F16", "F32"]))
@click.option("--seq-len", default=128, type=int, help="Sequence length for ONNX")
@click.option("--opset", default=17, type=int, help="ONNX opset")
@click.option("--ctx", "n_ctx", default=2048, type=int, help="Context length for GGUF")
@click.option("--soul-name", default=None, help="Slo name")
@click.option("--metadata", multiple=True, help="Metadata KEY=VALUE")
def model_export(checkpoint, output, fmt, quantize, seq_len, opset, n_ctx, soul_name, metadata):
    from commands.models import cmd_export_cli
    args = _ns(
        model=checkpoint, output=output, format=fmt, quantization=quantize,
        seq_len=seq_len, opset=opset, n_ctx=n_ctx, soul_name=soul_name,
        metadata=list(metadata) or None,
    )
    cmd_export_cli(args)


@model.command("benchmark", help="Run performance benchmarks")
@click.option("--checkpoint", "-m", default="gpt2", help="Model to benchmark")
@click.option("--device", "-d", type=click.Choice(["auto", "cpu", "cuda", "mps"]), default="auto")
@click.option("--test", "-t", type=click.Choice(["all", "latency", "throughput"]), default="all")
@click.option("--runs", "-r", default=10, type=int, help="Number of runs")
@click.option("--tokens", "-k", default=50, type=int, help="Max new tokens")
@click.option("--prompt", "-p", default="The quick brown fox jumps over the lazy dog", help="Test prompt")
@click.pass_context
def model_benchmark(ctx, checkpoint, device, test, runs, tokens, prompt):
    from commands.models import cmd_benchmark
    args = _ns(model=checkpoint, device=device, test=test, runs=runs, tokens=tokens, prompt=prompt,
              json_output=ctx.obj.get("json"))
    cmd_benchmark(args)


@model.command("compare", help="Compare models or benchmarks")
@click.pass_context
def model_compare(ctx):
    from commands.models import _cmd_models_compare
    _cmd_models_compare(_ns(json_output=ctx.obj.get("json")))


# ═══════════════════════════════════════════════════════════════════════
# dataset  — list, stats, search, import, export, validate
# ═══════════════════════════════════════════════════════════════════════


@cli.group(help="List, import, export, and validate datasets")
@click.pass_context
def dataset(ctx):
    pass


@dataset.command("list", help="List available datasets")
@click.pass_context
def dataset_list(ctx):
    from commands.data import cmd_datasets
    cmd_datasets(_ns(json_output=ctx.obj.get("json")))


@dataset.command("stats", help="Show dataset statistics")
@click.argument("name")
@click.pass_context
def dataset_stats(ctx, name):
    from commands.data import cmd_dataset_stats
    args = _ns(name=name, json_output=ctx.obj.get("json"))
    cmd_dataset_stats(args)


@dataset.command("search", help="Search online datasets")
@click.argument("query")
@click.option("--limit", "-n", default=10, type=int, help="Max results")
@click.option("--source", type=click.Choice(["hf", "github"]), default="hf")
@click.pass_context
def dataset_search(ctx, query, limit, source):
    from commands.data import cmd_dataset_search
    args = _ns(query=query, limit=limit, source=source, json_output=ctx.obj.get("json"))
    cmd_dataset_search(args)


@dataset.command("import", help="Import dataset from various sources")
@click.argument("source", type=click.Choice(["github", "hf", "url", "local"]))
@click.argument("identifier")
@click.argument("name", required=False)
def dataset_import(source, identifier, name):
    from commands.data import cmd_dataset_import
    args = _ns(**({"url": identifier} if source in ("github", "url") else {"dataset_id": identifier}), name=name)
    cmd_dataset_import(args, source)


@dataset.command("export", help="Export dataset to zip")
@click.argument("name")
@click.option("--output", "-o", help="Output zip file")
def dataset_export(name, output):
    from commands.data import cmd_dataset_export
    args = _ns(name=name, output=output)
    cmd_dataset_export(args)


@dataset.command("validate", help="Validate dataset file")
@click.argument("path")
def dataset_validate(path):
    from commands.data import cmd_data_tool
    cmd_data_tool(_ns(path=path), "validate")


@dataset.command("info", help="Show file or directory statistics")
@click.argument("path")
def dataset_info(path):
    from commands.data import cmd_data_tool
    cmd_data_tool(_ns(path=path), "stats")


# ═══════════════════════════════════════════════════════════════════════
# train  — start, quick, auto, self, eval, monitor, rlhf, demo, cloud
# ═══════════════════════════════════════════════════════════════════════


@cli.group(help="Train, evaluate, and monitor models")
@click.pass_context
def train(ctx):
    pass


@train.command("start", help="Full training pipeline")
@click.option("--dataset", default="shakespeare", help="Dataset name")
@click.option("--epochs", default=3, type=int, help="Training epochs")
@click.option("--batch-size", default=32, type=int, help="Batch size")
@click.option("--lr", default=0.01, type=float, help="Learning rate")
@click.option("--api", is_flag=True, help="Use API training")
@click.option("--resume", default=None, help="Resume from checkpoint")
@click.option("--resume-latest", is_flag=True, help="Resume latest")
@click.option("--save-stem", default=None, help="Output filename stem")
@click.pass_context
def train_start(ctx, dataset, epochs, batch_size, lr, api, resume, resume_latest, save_stem):
    from commands.train import cmd_train
    kwargs = dict(
        dataset=dataset, epochs=epochs, batch_size=batch_size, lr=lr,
        api=api, resume=resume, resume_latest=resume_latest,
        save_stem=save_stem,
        host=ctx.obj["host"], port=ctx.obj["port"], config=ctx.obj["config"],
    )
    cmd_train(_ns(**kwargs))


@train.command("native", help="Train a SloNet model from scratch (.soul checkpoints)")
@click.option("--dataset", default="datasets/tinyshakespeare/input.txt", help="Corpus file or dataset name")
@click.option("--steps", default=None, type=int, help="Max training steps (default: epoch budget)")
@click.option("--embed", default=64, type=int, help="Embedding dimension")
@click.option("--layers", default=2, type=int, help="Transformer layers")
@click.option("--heads", default=4, type=int, help="Attention heads")
@click.option("--block", default=128, type=int, help="Context block size")
@click.option("--batch", default=16, type=int, help="Batch size")
@click.option("--epochs", default=1, type=int, help="Training epochs")
@click.option("--lr", default=3e-3, type=float, help="Learning rate")
@click.option("--weight-decay", default=0.01, type=float, help="Weight decay")
@click.option("--scheduler", default="cosine", help="LR scheduler (cosine/linear/constant)")
@click.option("--warmup", default=100, type=int, help="Warmup steps")
@click.option("--min-lr", default=1e-5, type=float, help="Minimum learning rate")
@click.option("--grad-norm", default=1.0, type=float, help="Max gradient norm (0 disables clipping)")
@click.option("--dropout", default=0.1, type=float, help="Dropout")
@click.option("--checkpoint-dir", default="models/slonet-native", help="Checkpoint directory")
@click.option("--checkpoint-interval", default=500, type=int, help="Checkpoint interval (steps)")
@click.option("--max-checkpoints", default=3, type=int, help="Max checkpoints to keep")
@click.option("--save-best-only", is_flag=True, help="Only keep best-eval checkpoints")
@click.option("--eval-interval", default=250, type=int, help="Eval interval (steps)")
@click.option("--log-interval", default=50, type=int, help="Progress log interval (steps)")
@click.option("--soul-name", default="sloughgpt-native", help="Soul name for the checkpoint")
@click.option("--save-stem", default=None, help="Output filename stem (default: soul name)")
@click.option("--save-format", default="soul", type=click.Choice(["soul", "sou", "npz"]), help="DEPRECATED — ignored; SloughGPTTrainer.save() always writes .soul")
@click.option("--resume", default=None, help="Resume from a .soul/.npz checkpoint path")
@click.option("--resume-latest", is_flag=True, help="Resume from latest checkpoint in --checkpoint-dir")
@click.option("--device", default="cpu", help="Device (cpu/auto)")
@click.option("--tokenizer", default="char", type=click.Choice(["char", "token-tree"]), help="Tokenization strategy for the corpus")
@click.option("--token-vocab-size", default=512, type=int, help="Token tree vocabulary size (token-tree tokenizer)")
@click.option("--prompt", default=None, help="Generate a sample from this prompt after training")
@click.pass_context
def train_native(ctx, **kwargs):
    from commands.train import cmd_train_native
    kwargs["host"] = ctx.obj["host"]
    kwargs["port"] = ctx.obj["port"]
    cmd_train_native(_ns(**kwargs))


@train.command("quick", help="Smoke test: train briefly and generate")
@click.option("--dataset", "-d", default="datasets/shakespeare/input.txt", help="Corpus file")
@click.option("--prompt", default="The king", help="Generation prompt")
@click.option("--epochs", default=1, type=int, help="Training epochs")
@click.option("--steps", default=100, type=int, help="Max steps")
@click.option("--embed", default=128, type=int, help="Embedding size")
@click.option("--layers", default=4, type=int, help="Transformer layers")
@click.option("--heads", default=4, type=int, help="Attention heads")
@click.option("--block", default=128, type=int, help="Context length")
@click.option("--batch", default=16, type=int, help="Batch size")
@click.option("--lr", default=1e-3, type=float, help="Learning rate")
@click.option("--max-tokens", default=100, type=int, help="Generated tokens")
@click.option("--temperature", default=0.8, type=float, help="Temperature")
@click.option("--output", default="models/quick.soul", help="Output path")
@click.option("--no-optimize", is_flag=True, help="Disable optimizations")
@click.option("--soul-name", default="SloughGPT-Quick", help="Slo name")
@click.option("--datasets", help="Comma-separated datasets (overrides --dataset)")
@click.option("--ratios", help="Comma-separated dataset ratios")
@click.option("--preset", type=click.Choice(["tiny", "small", "medium", "large"]), help="Model preset")
@click.pass_context
def train_quick(ctx, **kwargs):
    from commands.train import cmd_quick
    kwargs["host"] = ctx.obj["host"]
    kwargs["port"] = ctx.obj["port"]
    cmd_quick(_ns(**kwargs))


@train.command("auto", help="Control auto-training via API")
@click.argument("action", type=click.Choice(["start", "stop", "status"]))
@click.option("--teacher", default="gpt2", help="Teacher model")
@click.option("--temperature", default=0.8, type=float, help="Temperature")
@click.option("--steps", default=1000, type=int, help="Max steps")
@click.pass_context
def train_auto(ctx, action, teacher, temperature, steps):
    from commands.train import _cmd_autotrain
    args = _ns(
        action=action, teacher=teacher, temperature=temperature,
        steps=steps, host=ctx.obj["host"], port=ctx.obj["port"],
    )
    _cmd_autotrain(args)


@train.command(name="self", help="Model talks to itself")
@click.option("--steps", default=1000, type=int, help="Training steps")
@click.option("--model", default="gpt2", help="Teacher model")
@click.option("--temperature", default=0.8, type=float, help="Temperature")
@click.option("--max-tokens", default=50, type=int, help="Max tokens per generation")
@click.option("--seed", default="Hello", help="Starting text")
@click.option("--forever", is_flag=True, help="Run until Ctrl+C")
def train_self(steps, model, temperature, max_tokens, seed, forever):
    from commands.train import _cmd_self_train
    args = _ns(
        steps=steps, model=model, temperature=temperature,
        max_tokens=max_tokens, seed=seed, forever=forever,
    )
    _cmd_self_train(args)


@train.command("eval", help="Evaluate model perplexity")
@click.option("--checkpoint", default="models/sloughgpt.soul", help="Checkpoint path")
@click.option("--data", default="datasets/shakespeare/input.txt", help="Eval text")
@click.option("--benchmark", is_flag=True, help="Run benchmark")
def train_eval(checkpoint, data, benchmark):
    from commands.train import cmd_eval
    args = _ns(checkpoint=checkpoint, data=data, benchmark=benchmark)
    cmd_eval(args)


@train.command("monitor", help="Monitor training jobs (delegates to dashboard)")
@click.option("--watch", is_flag=True, help="Continuous watch (ignored — always live)")
@click.option("--interval", default=2, type=int, help="Refresh interval (s)")
@click.pass_context
def train_monitor(ctx, watch, interval):
    from commands.monitor import monitor as _monitor_cmd
    ctx.invoke(_monitor_cmd, interval=float(interval), host=ctx.obj["host"], port=ctx.obj["port"], output_json=False, no_clear=False)


@train.command("rlhf", help="Run RLHF demo")
@click.option("--steps", default=20, type=int, help="PPO steps")
def train_rlhf(steps):
    from commands.train import cmd_rlhf
    args = _ns(steps=steps)
    cmd_rlhf(args)


@train.command("demo", help="Run system demos (RAG, KG, EWC)")
@click.option("--component", type=click.Choice(["all", "rag", "kg", "ewc", "inference"]), default="all")
def train_demo(component):
    from commands.train import cmd_demo
    args = _ns(component=component)
    cmd_demo(args)


@train.command("cloud", help="Setup Pinecone vector store")
@click.option("--api-key", help="Pinecone API key")
@click.option("--index", default="sloughgpt", help="Index name")
@click.option("--dimension", default=768, type=int, help="Vector dimension")
@click.option("--environment", default="us-east-1", help="Pinecone environment")
def train_cloud(api_key, index, dimension, environment):
    from commands.train import cmd_cloud_setup
    args = _ns(api_key=api_key, index=index, dimension=dimension, environment=environment)
    cmd_cloud_setup(args)


@train.command("embed", help="Train a text embedder on your corpus (no downloads)")
@click.option("--corpus", default=None, help="Text file or directory to train on (default: knowledge + chat history)")
@click.option("--epochs", default=20, type=int, help="Training epochs")
@click.option("--lr", default=3e-4, type=float, help="Learning rate")
@click.option("--batch-size", default=32, type=int, help="Batch size")
@click.option("--embed-dim", default=384, type=int, help="Embedding dimension")
@click.option("--vocab-size", default=4096, type=int, help="Max vocabulary size")
@click.option("--output", default=None, help="Output checkpoint path")
@click.option("--test", default=None, help="Test: embed a query string and print top matches")
def train_embed(corpus, epochs, lr, batch_size, embed_dim, vocab_size, output, test):
    """Train a text embedder on your own data using contrastive learning.

    \b
    Examples:
      sloughgpt train embed                          # train on knowledge + chat history
      sloughgpt train embed --corpus datasets/       # train on a directory of text files
      sloughgpt train embed --corpus my_corpus.txt   # train on a single file
      sloughgpt train embed --test "neural networks" # embed a test query
    """
    from commands.train import cmd_train_embed
    args = _ns(
        corpus=corpus, epochs=epochs, lr=lr, batch_size=batch_size,
        embed_dim=embed_dim, vocab_size=vocab_size, output=output, test=test,
    )
    cmd_train_embed(args)


# ═══════════════════════════════════════════════════════════════════════
# distill — knowledge distillation from teacher → student
# ═══════════════════════════════════════════════════════════════════════


@train.command("distill", help="Distill a teacher model into a smaller student")
@click.argument("text_source", required=False, default=None)
@click.option("--file", "-f", default=None, help="Text file to train on")
@click.option("--epochs", default=10, type=int, help="Training epochs")
@click.option("--lr", default=3e-4, type=float, help="Learning rate")
@click.option("--batch-size", default=8, type=int, help="Batch size")
@click.option("--n-embed", default=128, type=int, help="Student embedding size")
@click.option("--n-layer", default=4, type=int, help="Student layers")
@click.option("--n-head", default=4, type=int, help="Student attention heads")
@click.option("--block-size", default=128, type=int, help="Context length")
@click.option("--temperature", default=4.0, type=float, help="Distillation temperature")
@click.option("--dropout", default=0.1, type=float, help="Dropout rate")
@click.option("--checkpoint-dir", default="models/auto-training", help="Save directory")
@click.option("--log-interval", default=10, type=int, help="Log every N steps")
@click.option("--preset", type=click.Choice(["tiny", "small", "medium"]), help="Architecture preset")
@click.option("--api", is_flag=True, help="Use server API instead of local")
@click.option("--json", "json_output", is_flag=True, help="JSON output")
@click.option("--resume", default=None, help="Resume from checkpoint path (.soul file)")
@click.pass_context
def train_distill(ctx, text_source, file, epochs, lr, batch_size, n_embed, n_layer,
                  n_head, block_size, temperature, dropout, checkpoint_dir,
                  log_interval, preset, api, json_output, resume):
    """Distill GPT-2 into a smaller, faster student model.

    \b
    Examples:
      sloughgpt train distill datasets/shakespeare/input.txt
      sloughgpt train distill -f my_book.txt --epochs 20 --preset small
      sloughgpt train distill datasets/shakespeare/input.txt --api
      sloughgpt train distill datasets/shakespeare/input.txt --n-embed 64 --n-layer 2
      sloughgpt train distill datasets/shakespeare/input.txt --resume models/auto-training/checkpoint.soul
    """
    from commands.train import cmd_distill
    args = _ns(
        text_source=text_source, file=file, epochs=epochs, lr=lr,
        batch_size=batch_size, n_embed=n_embed, n_layer=n_layer,
        n_head=n_head, block_size=block_size, temperature=temperature,
        dropout=dropout, checkpoint_dir=checkpoint_dir,
        log_interval=log_interval, preset=preset, api=api,
        json_output=json_output, host=ctx.obj["host"], port=ctx.obj["port"],
        resume=resume,
    )
    cmd_distill(args)


@train.command("from-sessions", help="Train on your API chat logs (sessions + response logs)")
@click.option("--epochs", default=5, type=int, help="Training epochs")
@click.option("--lr", default=3e-4, type=float, help="Learning rate")
@click.option("--batch-size", default=8, type=int, help="Batch size")
@click.option("--n-embed", default=128, type=int, help="Embedding dimension")
@click.option("--n-layer", default=4, type=int, help="Transformer layers")
@click.option("--n-head", default=4, type=int, help="Attention heads")
@click.option("--block-size", default=128, type=int, help="Context block size")
@click.option("--dropout", default=0.1, type=float, help="Dropout rate")
@click.option("--soul-name", default="chat-trained", help="Name for the trained soul")
@click.option("--min-quality", default=2.0, type=float, help="Min pair quality (0-5)")
@click.option("--max-pairs", default=500, type=int, help="Max training pairs to use")
@click.option("--session-ids", default=None, help="Comma-separated session IDs (default: all)")
@click.option("--load", "auto_load", is_flag=True, help="Auto-load checkpoint into chat after training")
@click.option("--json", "json_output", is_flag=True, help="JSON output")
@click.pass_context
def train_from_sessions(ctx, epochs, lr, batch_size, n_embed, n_layer, n_head,
                        block_size, dropout, soul_name, min_quality, max_pairs,
                        session_ids, auto_load, json_output):
    """Train a model on your API chat logs.

    \b
    Examples:
      sloughgpt train from-sessions                          # train with defaults
      sloughgpt train from-sessions --epochs 10 --lr 1e-3    # tune hyperparams
      sloughgpt train from-sessions --load                   # train + load into chat
      sloughgpt train from-sessions --max-pairs 1000         # use more data
    """
    from commands.train import cmd_train_from_sessions
    args = _ns(
        epochs=epochs, lr=lr, batch_size=batch_size,
        n_embed=n_embed, n_layer=n_layer, n_head=n_head,
        block_size=block_size, dropout=dropout,
        soul_name=soul_name, min_quality=min_quality,
        max_pairs=max_pairs, session_ids=session_ids,
        auto_load=auto_load, json_output=json_output,
        host=ctx.obj["host"], port=ctx.obj["port"],
    )
    cmd_train_from_sessions(args)


# ═══════════════════════════════════════════════════════════════════════
# token-tree — train, encode, decode, and query a tree tokenizer
# ═══════════════════════════════════════════════════════════════════════


@cli.group("token-tree", help="Train, encode, decode, and query a tree tokenizer")
@click.pass_context
def token_tree(ctx):
    pass


@token_tree.command("train", help="Train a tree tokenizer from a corpus and save it")
@click.option("--corpus", "-c", default="datasets/tinyshakespeare/input.txt", help="Corpus file or dataset name")
@click.option("--vocab-size", "-v", default=512, type=int, help="Target vocabulary size")
@click.option("--embed-dim", "-e", default=64, type=int, help="Embedding dimension (0 disables embeddings)")
@click.option("--min-freq", default=2, type=int, help="Minimum pair frequency to merge")
@click.option("--output", "-o", default="models/slonet-native/token_tree", help="Save base path")
@click.pass_context
def token_tree_train(ctx, **kwargs):
    from commands.token_tree import cmd_token_tree_train
    cmd_token_tree_train(_ns(**kwargs))


@token_tree.command("encode", help="Encode text into token ids (reads stdin when no --text)")
@click.option("--tree", "-t", default="models/slonet-native/token_tree", help="Saved tree base path")
@click.option("--text", default=None, help="Text to encode")
@click.pass_context
def token_tree_encode(ctx, **kwargs):
    from commands.token_tree import cmd_token_tree_encode
    cmd_token_tree_encode(_ns(**kwargs))


@token_tree.command("decode", help="Decode comma-separated token ids back to text")
@click.option("--tree", "-t", default="models/slonet-native/token_tree", help="Saved tree base path")
@click.argument("ids")
@click.pass_context
def token_tree_decode(ctx, ids, **kwargs):
    from commands.token_tree import cmd_token_tree_decode
    kwargs["ids"] = ids
    cmd_token_tree_decode(_ns(**kwargs))


@token_tree.command("stats", help="Show training statistics for a saved tree")
@click.option("--tree", "-t", default="models/slonet-native/token_tree", help="Saved tree base path")
@click.pass_context
def token_tree_stats(ctx, **kwargs):
    from commands.token_tree import cmd_token_tree_stats
    cmd_token_tree_stats(_ns(**kwargs))


@token_tree.command("similar", help="Find nearest-neighbor tokens via generated embeddings")
@click.option("--tree", "-t", default="models/slonet-native/token_tree", help="Saved tree base path")
@click.option("--top-k", "-k", default=5, type=int, help="Number of results")
@click.argument("token")
@click.pass_context
def token_tree_similar(ctx, token, **kwargs):
    from commands.token_tree import cmd_token_tree_similar
    kwargs["token"] = token
    cmd_token_tree_similar(_ns(**kwargs))


@token_tree.command("lineage", help="Render a token's merge lineage down to its leaves")
@click.option("--tree", "-t", default="models/slonet-native/token_tree", help="Saved tree base path")
@click.argument("token")
@click.pass_context
def token_tree_lineage(ctx, token, **kwargs):
    from commands.token_tree import cmd_token_tree_lineage
    kwargs["token"] = token
    cmd_token_tree_lineage(_ns(**kwargs))


@token_tree.command("vocab", help="List a paged slice of the vocabulary with flags")
@click.option("--tree", "-t", default="models/slonet-native/token_tree", help="Saved tree base path")
@click.option("--offset", default=0, type=int, help="Number of leading entries to skip")
@click.option("--limit", "-n", default=50, type=int, help="Maximum entries to print (0 = no limit)")
@click.pass_context
def token_tree_vocab(ctx, **kwargs):
    from commands.token_tree import cmd_token_tree_vocab
    cmd_token_tree_vocab(_ns(**kwargs))


@token_tree.command("embedding", help="Inspect a token's generated embedding vector")
@click.option("--tree", "-t", default="models/slonet-native/token_tree", help="Saved tree base path")
@click.option("--top-k", "-k", default=8, type=int, help="Largest-magnitude dimensions to show")
@click.argument("token")
@click.pass_context
def token_tree_embedding(ctx, token, **kwargs):
    from commands.token_tree import cmd_token_tree_embedding
    kwargs["token"] = token
    cmd_token_tree_embedding(_ns(**kwargs))


@token_tree.command("path", help="Trace the greedy trie walk over text (reads stdin when no --text)")
@click.option("--tree", "-t", default="models/slonet-native/token_tree", help="Saved tree base path")
@click.option("--text", default=None, help="Text to trace")
@click.pass_context
def token_tree_path(ctx, **kwargs):
    from commands.token_tree import cmd_token_tree_path
    cmd_token_tree_path(_ns(**kwargs))


@token_tree.command("matrix", help="Summarize the full embedding matrix")
@click.option("--tree", "-t", default="models/slonet-native/token_tree", help="Saved tree base path")
@click.option("--top-k", "-k", default=8, type=int, help="Most/least energetic tokens to show")
@click.pass_context
def token_tree_matrix(ctx, **kwargs):
    from commands.token_tree import cmd_token_tree_matrix
    cmd_token_tree_matrix(_ns(**kwargs))


@token_tree.command("compare", help="Diff two saved token trees by name")
@click.option("--a", "-a", "a_name", required=True, help="First saved tree name")
@click.option("--b", "-b", "b_name", required=True, help="Second saved tree name")
@click.option("--top-k", "-k", default=10, type=int, help="Shared/exclusive token examples per side")
@click.pass_context
def token_tree_compare(ctx, a_name, b_name, top_k):
    from commands.token_tree import cmd_token_tree_compare
    cmd_token_tree_compare(_ns(a=a_name, b=b_name, top_n=top_k))


@token_tree.command("merges", help="List the most frequent BPE merge rules of a saved tree")
@click.option("--tree", "-t", default="models/slonet-native/token_tree", help="Saved tree base path")
@click.option("--top-n", "-n", default=20, type=int, help="Maximum merge rules to show")
@click.option("--query", "-q", default="", help="Filter rules whose parts contain this substring")
@click.pass_context
def token_tree_merges(ctx, **kwargs):
    from commands.token_tree import cmd_token_tree_merges
    cmd_token_tree_merges(_ns(**kwargs))


@token_tree.command("saved", help="List saved token trees")
@click.pass_context
def token_tree_saved(ctx):
    from commands.token_tree import cmd_token_tree_saved
    cmd_token_tree_saved(_ns())


@token_tree.command("save", help="Save the current tree (or --tree path) under a name")
@click.option("--name", "-n", "name", required=True, help="Name to save the tree under")
@click.option("--tree", "-t", default=None, help="Optional saved tree base path to adopt first")
@click.pass_context
def token_tree_save(ctx, name, **kwargs):
    from commands.token_tree import cmd_token_tree_save
    cmd_token_tree_save(_ns(name=name, **kwargs))


@token_tree.command("load", help="Load a saved tree by name and make it current")
@click.argument("name")
@click.pass_context
def token_tree_load(ctx, name):
    from commands.token_tree import cmd_token_tree_load
    cmd_token_tree_load(_ns(name=name))


@token_tree.command("delete", help="Delete a saved token tree by name")
@click.argument("name")
@click.option("--dry-run", is_flag=True, help="Show what would be deleted without deleting")
@click.pass_context
def token_tree_delete(ctx, name, dry_run):
    from commands.token_tree import cmd_token_tree_delete
    if dry_run:
        log.info(f"Would delete token tree: {name}")
        return
    cmd_token_tree_delete(_ns(name=name))


# ═══════════════════════════════════════════════════════════════════════
# checkpoint — list, load, delete training checkpoints
# ═══════════════════════════════════════════════════════════════════════


@cli.group(help="List, load, and delete training checkpoints")
def checkpoint():
    pass


@checkpoint.command("list", help="List all training checkpoints")
@click.option("--sort", type=click.Choice(["date", "size", "name"]), default="date", help="Sort order")
@click.option("--json", "json_output", is_flag=True, help="JSON output")
@click.pass_context
def checkpoint_list(ctx, sort, json_output):
    """List all saved training checkpoints.

    \b
    Examples:
      sloughgpt checkpoint list
      sloughgpt checkpoint list --sort size
      sloughgpt checkpoint list --json
    """
    import requests
    base_url = f"http://{ctx.obj['host']}:{ctx.obj['port']}"
    resp = requests.get(f"{base_url}/training/checkpoints", timeout=10)
    if resp.status_code != 200:
        log.error(f"Failed to list checkpoints: {resp.text}")
        sys.exit(1)
    checkpoints = resp.json()
    if not checkpoints:
        log.info("No checkpoints found")
        return

    if json_output:
        log.json(checkpoints)
        return

    log.header(f"Training Checkpoints ({len(checkpoints)})")
    rows = []
    for cp in checkpoints:
        name = cp.get("name", "unknown")
        size = cp.get("size_mb", 0)
        traits = cp.get("traits", {})
        trait_str = ", ".join(f"{k}={v:.2f}" for k, v in traits.items() if v != 0.5) if traits else ""
        rows.append([name, f"{size:.1f} MB", trait_str or "-"])
    log.table(["Name", "Size", "Traits"], rows)


@checkpoint.command("load", help="Load a checkpoint into the model")
@click.argument("name")
@click.pass_context
def checkpoint_load(ctx, name):
    """Load a training checkpoint into the active model.

    \b
    Example:
      sloughgpt checkpoint load my-checkpoint.soul
    """
    import requests
    base_url = f"http://{ctx.obj['host']}:{ctx.obj['port']}"
    resp = requests.post(f"{base_url}/training/checkpoints/{name}/load", timeout=30)
    if resp.status_code == 200:
        data = resp.json()
        log.success(f"Loaded checkpoint: {name}")
        for k, v in data.items():
            if k not in ("status",):
                log.key_value(k, str(v))
    else:
        log.error(f"Failed to load: {resp.text}")
        sys.exit(1)


@checkpoint.command("delete", help="Delete a training checkpoint")
@click.argument("name")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
@click.option("--dry-run", is_flag=True, help="Show what would be deleted without deleting")
@click.pass_context
def checkpoint_delete(ctx, name, yes, dry_run):
    """Delete a training checkpoint.

    \b
    Example:
      sloughgpt checkpoint delete my-checkpoint.soul
    """
    if not yes and not dry_run:
        confirm(f"Delete checkpoint '{name}'?", abort=True)
    import requests
    base_url = f"http://{ctx.obj['host']}:{ctx.obj['port']}"
    if dry_run:
        log.info(f"Would delete: {name}")
        return
    resp = requests.delete(f"{base_url}/training/checkpoints/{name}", timeout=10)
    if resp.status_code == 200:
        log.success(f"Deleted: {name}")
    else:
        log.error(f"Failed to delete: {resp.text}")
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════
# knowledge  — search, duplicates, categorize, gaps, ingest
# ═══════════════════════════════════════════════════════════════════════


@cli.group(help="Semantic knowledge operations — search, dedup, categorize, gaps")
def knowledge():
    pass


@knowledge.command("search", help="Search codebase with natural language")
@click.argument("query")
@click.option("--path", default=".", help="Directory to search")
@click.option("--top-k", default=10, type=int, help="Max results")
@click.option("--extensions", default=None, help="Comma-separated file extensions")
@click.pass_context
def knowledge_search(ctx, query, path, top_k, extensions):
    """Search your codebase using natural language.

    \b
    Examples:
      sloughgpt knowledge search "how does embedding work"
      sloughgpt knowledge search "training loop" --path packages/core-py
      sloughgpt knowledge search "error handling" --extensions py,ts
    """
    import requests
    exts = extensions.split(",") if extensions else None
    r = requests.post(f"http://{ctx.obj['host']}:{ctx.obj['port']}/knowledge/search-files",
                      json={"query": query, "path": path, "top_k": top_k, "extensions": exts})
    if r.status_code != 200:
        log.error(f"Search failed: {r.text}")
        return
    data = r.json()
    log.header(f"Found {len(data['results'])} results (indexed {data['indexed_files']} files)")
    for i, res in enumerate(data["results"], 1):
        log.info(f"[{res['score']:.3f}] {res['path']}:{res['line']}")
        snippet = res['snippet'].replace('\n', ' ')[:100]
        log.info(f"  {snippet}")
        log.blank()


@knowledge.command("dedup", help="Check for duplicate knowledge")
@click.argument("content")
@click.option("--threshold", default=0.85, type=float, help="Similarity threshold")
@click.pass_context
def knowledge_dedup(ctx, content, threshold):
    """Check if content already exists in the knowledge base.

    \b
    Example:
      sloughgpt knowledge dedup "neural networks learn from data"
    """
    import requests
    r = requests.post(f"http://{ctx.obj['host']}:{ctx.obj['port']}/knowledge/check-duplicate",
                      json={"content": content, "threshold": threshold})
    if r.status_code != 200:
        log.error(f"Check failed: {r.text}")
        return
    data = r.json()
    if data["is_duplicate"]:
        log.warning(f"DUPLICATE (score: {data['score']:.3f})")
        log.info(f"  Existing: {data['best_match'][:100]}")
    else:
        log.success(f"Unique (best match score: {data['score']:.3f})")


@knowledge.command("categorize", help="Auto-categorize content")
@click.argument("content")
@click.pass_context
def knowledge_categorize(ctx, content):
    """Auto-assign a topic to content based on existing categories.

    \b
    Example:
      sloughgpt knowledge categorize "gradient descent optimizes loss"
    """
    import requests
    r = requests.post(f"http://{ctx.obj['host']}:{ctx.obj['port']}/knowledge/categorize",
                      json={"content": content})
    if r.status_code != 200:
        log.error(f"Categorize failed: {r.text}")
        return
    data = r.json()
    log.success(f"Topic: {data['topic']}")
    if data["suggestions"]:
        log.info("Suggestions:")
        for s in data["suggestions"]:
            log.info(f"  {s['topic']} ({s['score']:.3f})")


@knowledge.command("gaps", help="Find knowledge gaps")
@click.pass_context
def knowledge_gaps(ctx):
    """Show under-represented topics in your knowledge base."""
    import requests
    r = requests.get(f"http://{ctx.obj['host']}:{ctx.obj['port']}/knowledge/gaps")
    if r.status_code != 200:
        log.error(f"Gaps failed: {r.text}")
        return
    data = r.json()
    log.header(f"Knowledge gaps ({data['total_facts']} facts, {len(data['topics'])} topics)")
    if data["gaps"]:
        for g in data["gaps"]:
            log.info(f"  {g['topic']}: {g['suggestion']}")
    else:
        log.success("No significant gaps found")


@knowledge.command("ingest", help="Bulk ingest texts with dedup")
@click.argument("texts", nargs=-1)
@click.option("--topic", default="imported", help="Topic tag")
@click.option("--file", "file_path", default=None, help="Read texts from file (one per line)")
@click.pass_context
def knowledge_ingest(ctx, texts, topic, file_path):
    """Bulk ingest texts with automatic deduplication.

    \b
    Examples:
      sloughgpt knowledge ingest "fact 1" "fact 2" "fact 3"
      sloughgpt knowledge ingest --file facts.txt --topic ml
    """
    import requests
    items = list(texts)
    if file_path:
        with open(file_path) as f:
            items.extend(line.strip() for line in f if line.strip())
    if not items:
        log.error("No texts to ingest")
        return
    timeout = ctx.obj.get("timeout", 10)
    r = requests.post(f"http://{ctx.obj['host']}:{ctx.obj['port']}/knowledge/bulk-ingest",
                      json={"items": items, "topic": topic}, timeout=timeout)
    if r.status_code != 200:
        log.error(f"Ingest failed: {r.text}")
        return
    data = r.json()
    log.success(f"Bulk ingest: {data['added']} added, {data['skipped']} skipped, {data['errors']} errors")


# ═══════════════════════════════════════════════════════════════════════
# experiment  — list, create, info, delete, metrics
# ═══════════════════════════════════════════════════════════════════════


@cli.group(help="ML experiment tracking — create, list, log metrics")
def experiment():
    pass


@experiment.command("list", help="List all experiments")
@click.pass_context
def experiment_list(ctx):
    import requests
    timeout = ctx.obj.get("timeout", 10)
    r = requests.get(f"http://{ctx.obj['host']}:{ctx.obj['port']}/experiments", timeout=timeout)
    if r.status_code != 200:
        log.error(f"Failed to list experiments: {r.text}")
        return
    data = r.json()
    exps = data.get("data", {}).get("experiments", [])
    if not exps:
        log.info("No experiments found")
        return
    if ctx.obj.get("json"):
        _output(ctx, {"experiments": exps})
    else:
        log.header("Experiments")
        for exp in exps:
            log.info(f"  {exp}")


@experiment.command("create", help="Create a new experiment")
@click.argument("name")
@click.pass_context
def experiment_create(ctx, name):
    import requests
    timeout = ctx.obj.get("timeout", 10)
    r = requests.post(f"http://{ctx.obj['host']}:{ctx.obj['port']}/experiments",
                      json={"name": name}, timeout=timeout)
    if r.status_code != 200:
        log.error(f"Failed to create experiment: {r.text}")
        return
    data = r.json().get("data", {})
    log.success(f"Created experiment: {data.get('id', name)}")


@experiment.command("info", help="Show experiment details")
@click.argument("experiment_id")
@click.pass_context
def experiment_info(ctx, experiment_id):
    import requests
    timeout = ctx.obj.get("timeout", 10)
    r = requests.get(f"http://{ctx.obj['host']}:{ctx.obj['port']}/experiments/{experiment_id}", timeout=timeout)
    if r.status_code != 200:
        log.error(f"Experiment not found: {r.text}")
        return
    data = r.json().get("data", {})
    if ctx.obj.get("json"):
        _output(ctx, data)
    else:
        log.header(f"Experiment: {experiment_id}")
        for k, v in data.items():
            log.key_value(k, str(v))


@experiment.command("delete", help="Delete an experiment")
@click.argument("experiment_id")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
@click.option("--dry-run", is_flag=True, help="Show what would be deleted")
@click.pass_context
def experiment_delete(ctx, experiment_id, yes, dry_run):
    import requests
    if dry_run:
        log.info(f"Would delete experiment: {experiment_id}")
        return
    if not yes:
        confirm(f"Delete experiment '{experiment_id}'?", abort=True)
    timeout = ctx.obj.get("timeout", 10)
    r = requests.delete(f"http://{ctx.obj['host']}:{ctx.obj['port']}/experiments/{experiment_id}", timeout=timeout)
    if r.status_code == 200:
        log.success(f"Deleted experiment: {experiment_id}")
    else:
        log.error(f"Failed to delete: {r.text}")


@experiment.command("metrics", help="Show experiment metrics")
@click.argument("experiment_id")
@click.pass_context
def experiment_metrics(ctx, experiment_id):
    import requests
    timeout = ctx.obj.get("timeout", 10)
    r = requests.get(f"http://{ctx.obj['host']}:{ctx.obj['port']}/experiments/{experiment_id}/data", timeout=timeout)
    if r.status_code != 200:
        log.error(f"Failed to get metrics: {r.text}")
        return
    data = r.json().get("data", {})
    if ctx.obj.get("json"):
        _output(ctx, data)
    else:
        log.header(f"Metrics: {experiment_id}")
        metrics = data.get("metrics", [])
        if not metrics:
            log.info("No metrics recorded yet")
            return
        for m in metrics[-20:]:
            log.info(f"  {m.get('step', '?')}: {m.get('key', '?')}={m.get('value', '?')}")


# ═══════════════════════════════════════════════════════════════════════
# error  — recent, grouped, trends, clear
# ═══════════════════════════════════════════════════════════════════════


@cli.group(help="Error monitoring — recent, grouped, trends, clear")
def error():
    pass


@error.command("recent", help="Show recent errors")
@click.option("--limit", "-n", default=20, type=int, help="Max errors to show")
@click.pass_context
def error_recent(ctx, limit):
    import requests
    timeout = ctx.obj.get("timeout", 10)
    r = requests.get(f"http://{ctx.obj['host']}:{ctx.obj['port']}/errors/recent?limit={limit}", timeout=timeout)
    if r.status_code != 200:
        log.error(f"Failed to fetch errors: {r.text}")
        return
    data = r.json().get("data", {})
    errors = data.get("errors", [])
    if not errors:
        log.info("No recent errors")
        return
    if ctx.obj.get("json"):
        _output(ctx, {"errors": errors})
    else:
        log.header(f"Recent Errors ({len(errors)})")
        for e in errors:
            ts = e.get("timestamp", "?")[:19]
            msg = e.get("message", "?")[:80]
            log.info(f"  [{ts}] {msg}")


@error.command("grouped", help="Show errors grouped by message")
@click.pass_context
def error_grouped(ctx):
    import requests
    timeout = ctx.obj.get("timeout", 10)
    r = requests.get(f"http://{ctx.obj['host']}:{ctx.obj['port']}/errors/grouped", timeout=timeout)
    if r.status_code != 200:
        log.error(f"Failed to fetch grouped errors: {r.text}")
        return
    data = r.json().get("data", {})
    groups = data.get("groups", [])
    if not groups:
        log.info("No errors grouped")
        return
    if ctx.obj.get("json"):
        _output(ctx, {"groups": groups})
    else:
        log.header(f"Error Groups ({len(groups)})")
        for g in groups:
            count = g.get("count", 0)
            msg = g.get("message", "?")[:70]
            log.info(f"  [{count}x] {msg}")


@error.command("trends", help="Show error trends (last 24h)")
@click.pass_context
def error_trends(ctx):
    import requests
    timeout = ctx.obj.get("timeout", 10)
    r = requests.get(f"http://{ctx.obj['host']}:{ctx.obj['port']}/errors/trends", timeout=timeout)
    if r.status_code != 200:
        log.error(f"Failed to fetch trends: {r.text}")
        return
    data = r.json().get("data", {})
    trends = data.get("trends", [])
    if not trends:
        log.info("No error trends")
        return
    if ctx.obj.get("json"):
        _output(ctx, {"trends": trends})
    else:
        log.header("Error Trends (24h)")
        for t in trends:
            hour = t.get("hour", "?")
            count = t.get("count", 0)
            bar = "#" * min(count, 40)
            log.info(f"  {hour}: {bar} ({count})")


@error.command("clear", help="Clear all errors")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
@click.option("--dry-run", is_flag=True, help="Show what would be cleared")
@click.pass_context
def error_clear(ctx, yes, dry_run):
    import requests
    if dry_run:
        log.info("Would clear all errors")
        return
    if not yes:
        confirm("Clear all errors?", abort=True)
    timeout = ctx.obj.get("timeout", 10)
    r = requests.delete(f"http://{ctx.obj['host']}:{ctx.obj['port']}/errors/clear", timeout=timeout)
    if r.status_code == 200:
        log.success("Errors cleared")
    else:
        log.error(f"Failed to clear: {r.text}")


@error.command("unread", help="Show unread error count")
@click.pass_context
def error_unread(ctx):
    import requests
    timeout = ctx.obj.get("timeout", 10)
    r = requests.get(f"http://{ctx.obj['host']}:{ctx.obj['port']}/errors/unread", timeout=timeout)
    if r.status_code != 200:
        log.error(f"Failed to fetch unread count: {r.text}")
        return
    data = r.json().get("data", {})
    count = data.get("count", 0)
    if ctx.obj.get("json"):
        _output(ctx, {"unread": count})
    else:
        log.info(f"Unread errors: {count}")


# ═══════════════════════════════════════════════════════════════════════
# memory  — stats, enable, disable, list, search, store, remember, consolidate, archive, clear
# ═══════════════════════════════════════════════════════════════════════


@cli.group(help="Inspect and manage the auto-memory layer (stats, search, store, consolidate, archive)")
def memory():
    pass


@memory.command("stats", help="Show memory statistics")
def memory_stats():
    from commands.memory import cmd_memory_stats
    cmd_memory_stats(_ns())


@memory.command("enable", help="Enable the memory layer at runtime")
def memory_enable():
    from commands.memory import cmd_memory_enable
    cmd_memory_enable(_ns(enabled=True))


@memory.command("disable", help="Disable the memory layer at runtime")
def memory_disable():
    from commands.memory import cmd_memory_enable
    cmd_memory_enable(_ns(enabled=False))


@memory.command("list", help="List stored memory items, most recent first")
@click.option("--limit", "-n", default=50, type=int, help="Max items to show")
def memory_list(limit):
    from commands.memory import cmd_memory_list
    cmd_memory_list(_ns(limit=limit))


@memory.command("search", help="Semantic-search stored memory")
@click.argument("query")
@click.option("--limit", "-n", default=5, type=int, help="Max results")
def memory_search(query, limit):
    from commands.memory import cmd_memory_search
    cmd_memory_search(_ns(query=query, limit=limit))


@memory.command("store", help="Persist one explicit fact")
@click.argument("content")
@click.option("--topic", default="manual", help="Topic label")
@click.option("--source", default="cli", help="Provenance label")
def memory_store(content, topic, source):
    from commands.memory import cmd_memory_store
    cmd_memory_store(_ns(content=content, topic=topic, source=source))


@memory.command("remember", help="Persist one completed turn (user + assistant)")
@click.argument("user_message")
@click.argument("assistant_response")
def memory_remember(user_message, assistant_response):
    from commands.memory import cmd_memory_remember
    cmd_memory_remember(_ns(
        user_message=user_message, assistant_response=assistant_response,
    ))


@memory.command("clear", help="Remove all stored memory")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
@click.option("--dry-run", is_flag=True, help="Show what would be cleared without clearing")
def memory_clear(yes, dry_run):
    from commands.memory import cmd_memory_clear
    if dry_run:
        log.info("Would clear all stored memory")
        return
    cmd_memory_clear(_ns(yes=yes))


@memory.command("consolidate", help="Merge near-duplicate facts, keeping the longest")
@click.option("--threshold", type=float, default=None,
              help="Min similarity for a merge (default from config)")
def memory_consolidate(threshold):
    from commands.memory import cmd_memory_consolidate
    cmd_memory_consolidate(_ns(threshold=threshold))


@memory.command("archive", help="Inspect or prune the task-backed provenance archive")
@click.option("--limit", "-n", default=10, type=int, help="Recent records to show (0 = none)")
@click.option("--prune-days", type=float, default=None,
              help="Retention window in days; delete older records")
def memory_archive(limit, prune_days):
    from commands.memory import cmd_memory_archive
    cmd_memory_archive(_ns(limit=limit, prune_days=prune_days))


# ═══════════════════════════════════════════════════════════════════════
# personality  — list, load, info, create, export
# ═══════════════════════════════════════════════════════════════════════


@cli.group(help="List, load, and manage .soul personality files")
def personality():
    pass


@personality.command("list", help="List built-in personalities")
def personality_list():
    from commands.models import _cmd_models_personalities
    _cmd_models_personalities(_ns())


@personality.command("load", help="Load soul via API")
@click.argument("path")
@click.pass_context
def personality_load(ctx, path):
    from commands.models import cmd_soul
    cmd_soul(_ns(load=path, host=ctx.obj["host"], port=ctx.obj["port"]))


@personality.command("info", help="Inspect soul file")
@click.argument("path")
def personality_info(path):
    from commands.models import cmd_soul
    cmd_soul(_ns(info=path))


@personality.command("create", help="Create new soul from checkpoint")
@click.option("--checkpoint", "-m", required=True, help="Weights path")
@click.option("--name", "-n", required=True, help="Soul name")
@click.option("--dataset", "-d", help="Dataset citation")
@click.option("--epochs", "-e", default=0, type=int, help="Epoch count")
@click.option("--lineage", default="nanogpt", help="Architecture label")
@click.option("--tags", default="", help="Comma-separated tags")
@click.option("--output", "-o", help="Output .soul path")
def personality_create(checkpoint, name, dataset, epochs, lineage, tags, output):
    from commands.models import cmd_soul
    args = _ns(
        create=output or f"models/{name}.soul", model=checkpoint,
        name=name, dataset=dataset, epochs=epochs, lineage=lineage, tags=tags,
    )
    cmd_soul(args)


# ═══════════════════════════════════════════════════════════════════════
# adapter  — list, info, merge, delete
# ═══════════════════════════════════════════════════════════════════════


@cli.group(help="Manage per-user LoRA adapters")
def adapter():
    pass


@adapter.command("list", help="List LoRA adapters")
def adapter_list():
    from commands.train import _cmd_user_adapters
    _cmd_user_adapters(_ns(action="list"))


@adapter.command("info", help="Show adapter info")
@click.argument("user")
def adapter_info(user):
    from commands.train import _cmd_user_adapters
    _cmd_user_adapters(_ns(action="info", user=user))


@adapter.command("merge", help="Merge adapters")
@click.option("--users", required=True, help="Comma-separated user IDs")
def adapter_merge(users):
    from commands.train import _cmd_user_adapters
    _cmd_user_adapters(_ns(action="merge", users=users))


@adapter.command("delete", help="Delete adapter")
@click.argument("user")
@click.option("--dry-run", is_flag=True, help="Show what would be deleted without deleting")
def adapter_delete(user, dry_run):
    from commands.train import _cmd_user_adapters
    if dry_run:
        log.info(f"Would delete adapter for user: {user}")
        return
    _cmd_user_adapters(_ns(action="delete", user=user))


# ═══════════════════════════════════════════════════════════════════════
# feedback  — export, prepare
# ═══════════════════════════════════════════════════════════════════════


@cli.group(help="Export and prepare feedback data")
def feedback():
    pass


@feedback.command("export", help="Export feedback data")
@click.option("--format", type=click.Choice(["jsonl", "dpo"]), default="jsonl")
@click.option("--output", default="data/training_feedback.jsonl")
def feedback_export(fmt, output):
    from commands.train import _cmd_feedback_export
    args = _ns(format=fmt, output=output)
    _cmd_feedback_export(args)


@feedback.command("prepare", help="Prepare training data from feedback")
@click.option("--format", type=click.Choice(["all", "dpo", "sft", "reward"]), default="all")
@click.option("--output")
@click.option("--stats-only", is_flag=True)
def feedback_prepare(fmt, output, stats_only):
    from commands.train import _cmd_feedback_train
    args = _ns(format=fmt, output=output, stats_only=stats_only)
    _cmd_feedback_train(args)


# ═══════════════════════════════════════════════════════════════════════
# agent  — list, create, execute, orchestrate
# ═══════════════════════════════════════════════════════════════════════


@cli.group(help="Manage and execute AI agents")
def agent():
    pass


@agent.command("list", help="List all agents")
@click.pass_context
def agent_list(ctx):
    import requests
    timeout = ctx.obj.get("timeout", 10)
    r = requests.get(f"http://{ctx.obj['host']}:{ctx.obj['port']}/agents", timeout=timeout)
    if r.status_code != 200:
        log.error(f"Failed to list agents: {r.text}")
        sys.exit(1)
    data = r.json()
    agents = data.get("data", data) if isinstance(data, dict) else data
    if isinstance(agents, dict):
        agents = agents.get("agents", [])
    if not agents:
        log.info("No agents found")
        return
    if ctx.obj.get("json"):
        _output(ctx, {"agents": agents})
    else:
        log.header("Agents")
        for a in agents:
            name = a.get("name", a.get("id", "?"))
            desc = a.get("description", "")[:60]
            log.info(f"  {name} — {desc}")


@agent.command("create", help="Create a new agent")
@click.argument("name")
@click.option("--description", "-d", default="", help="Agent description")
@click.option("--instructions", "-i", default="", help="System instructions")
@click.pass_context
def agent_create(ctx, name, description, instructions):
    import requests
    timeout = ctx.obj.get("timeout", 10)
    r = requests.post(f"http://{ctx.obj['host']}:{ctx.obj['port']}/agents",
                      json={"name": name, "description": description, "instructions": instructions},
                      timeout=timeout)
    if r.status_code != 200:
        log.error(f"Failed to create agent: {r.text}")
        sys.exit(1)
    log.success(f"Created agent: {name}")


@agent.command("execute", help="Execute a task with an agent")
@click.argument("agent_id")
@click.argument("request")
@click.pass_context
def agent_execute(ctx, agent_id, request):
    import requests
    timeout = ctx.obj.get("timeout", 30)
    r = requests.post(f"http://{ctx.obj['host']}:{ctx.obj['port']}/agents/{agent_id}/execute",
                      json={"request": request}, timeout=timeout)
    if r.status_code != 200:
        log.error(f"Execution failed: {r.text}")
        sys.exit(1)
    data = r.json()
    if ctx.obj.get("json"):
        _output(ctx, data)
    else:
        result = data.get("data", data).get("result", str(data))
        log.info(result)


@agent.command("orchestrate", help="Multi-agent orchestration")
@click.argument("goal")
@click.option("--context", "-c", default="", help="Additional context")
@click.option("--agents", default="", help="Comma-separated agent IDs")
@click.pass_context
def agent_orchestrate(ctx, goal, context, agents):
    import requests
    agent_ids = [a.strip() for a in agents.split(",") if a.strip()] if agents else []
    timeout = ctx.obj.get("timeout", 60)
    r = requests.post(f"http://{ctx.obj['host']}:{ctx.obj['port']}/agents/orchestrate",
                      json={"goal": goal, "context": context, "agent_ids": agent_ids},
                      timeout=timeout)
    if r.status_code != 200:
        log.error(f"Orchestration failed: {r.text}")
        sys.exit(1)
    data = r.json()
    if ctx.obj.get("json"):
        _output(ctx, data)
    else:
        result = data.get("data", data).get("result", str(data))
        log.info(result)


@agent.command("delete", help="Delete an agent")
@click.argument("agent_id")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
@click.option("--dry-run", is_flag=True, help="Show what would be deleted")
@click.pass_context
def agent_delete(ctx, agent_id, yes, dry_run):
    import requests
    if dry_run:
        log.info(f"Would delete agent: {agent_id}")
        return
    if not yes:
        confirm(f"Delete agent '{agent_id}'?", abort=True)
    timeout = ctx.obj.get("timeout", 10)
    r = requests.delete(f"http://{ctx.obj['host']}:{ctx.obj['port']}/agents/{agent_id}", timeout=timeout)
    if r.status_code == 200:
        log.success(f"Deleted agent: {agent_id}")
    else:
        log.error(f"Failed to delete: {r.text}")
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════
# session  — list, messages, search, inspector
# ═══════════════════════════════════════════════════════════════════════


@cli.group(help="Chat session management")
def session():
    pass


@session.command("list", help="List chat sessions")
@click.pass_context
def session_list(ctx):
    import requests
    timeout = ctx.obj.get("timeout", 10)
    r = requests.get(f"http://{ctx.obj['host']}:{ctx.obj['port']}/chat/sessions", timeout=timeout)
    if r.status_code != 200:
        log.error(f"Failed to list sessions: {r.text}")
        sys.exit(1)
    data = r.json()
    sessions = data if isinstance(data, list) else data.get("sessions", [])
    if not sessions:
        log.info("No sessions found")
        return
    if ctx.obj.get("json"):
        _output(ctx, {"sessions": sessions})
    else:
        log.header("Chat Sessions")
        for s in sessions:
            name = s.get("name", s.get("id", "?"))
            log.info(f"  {name}")


@session.command("messages", help="Show messages in a session")
@click.argument("session_id")
@click.option("--limit", "-n", default=20, type=int, help="Max messages")
@click.pass_context
def session_messages(ctx, session_id, limit):
    import requests
    timeout = ctx.obj.get("timeout", 10)
    r = requests.get(f"http://{ctx.obj['host']}:{ctx.obj['port']}/session/{session_id}/messages?limit={limit}", timeout=timeout)
    if r.status_code != 200:
        log.error(f"Failed to get messages: {r.text}")
        sys.exit(1)
    data = r.json()
    messages = data.get("messages", [])
    if not messages:
        log.info("No messages in session")
        return
    if ctx.obj.get("json"):
        _output(ctx, {"messages": messages})
    else:
        log.header(f"Session: {session_id}")
        for m in messages:
            role = m.get("role", "?")
            content = m.get("content", "")[:100]
            log.info(f"  [{role}] {content}")


@session.command("search", help="Search chat sessions")
@click.argument("query")
@click.option("--limit", "-n", default=10, type=int, help="Max results")
@click.pass_context
def session_search(ctx, query, limit):
    import requests
    timeout = ctx.obj.get("timeout", 10)
    r = requests.get(f"http://{ctx.obj['host']}:{ctx.obj['port']}/chat/sessions/search?q={query}&limit={limit}", timeout=timeout)
    if r.status_code != 200:
        log.error(f"Search failed: {r.text}")
        sys.exit(1)
    data = r.json()
    results = data if isinstance(data, list) else data.get("results", [])
    if not results:
        log.info("No matching sessions")
        return
    if ctx.obj.get("json"):
        _output(ctx, {"results": results})
    else:
        log.header(f"Search: {query}")
        for s in results:
            name = s.get("name", s.get("id", "?"))
            log.info(f"  {name}")


# ═══════════════════════════════════════════════════════════════════════
# tokenizer  — tokenize, detokenize, analyze, vocab, merges, train, stats
# ═══════════════════════════════════════════════════════════════════════


@cli.group(help="Tokenizer management and text analysis")
def tokenizer():
    pass


@tokenizer.command("tokenize", help="Tokenize text")
@click.argument("text")
@click.pass_context
def tokenizer_tokenize(ctx, text):
    import requests
    timeout = ctx.obj.get("timeout", 10)
    r = requests.post(f"http://{ctx.obj['host']}:{ctx.obj['port']}/tokenizer/tokenize",
                      json={"text": text}, timeout=timeout)
    if r.status_code != 200:
        log.error(f"Tokenize failed: {r.text}")
        sys.exit(1)
    data = r.json()
    if ctx.obj.get("json"):
        _output(ctx, data)
    else:
        tokens = data.get("data", data).get("tokens", [])
        log.info(f"Tokens: {tokens}")
        log.info(f"Count: {len(tokens)}")


@tokenizer.command("detokenize", help="Convert token IDs back to text")
@click.argument("ids")
@click.pass_context
def tokenizer_detokenize(ctx, ids):
    import requests
    id_list = [int(x.strip()) for x in ids.split(",")]
    timeout = ctx.obj.get("timeout", 10)
    r = requests.post(f"http://{ctx.obj['host']}:{ctx.obj['port']}/tokenizer/detokenize",
                      json={"ids": id_list}, timeout=timeout)
    if r.status_code != 200:
        log.error(f"Detokenize failed: {r.text}")
        sys.exit(1)
    data = r.json()
    text = data.get("data", data).get("text", "")
    log.info(f"Text: {text}")


@tokenizer.command("analyze", help="Analyze token distribution in text")
@click.argument("text")
@click.pass_context
def tokenizer_analyze(ctx, text):
    import requests
    timeout = ctx.obj.get("timeout", 10)
    r = requests.post(f"http://{ctx.obj['host']}:{ctx.obj['port']}/tokenizer/analyze",
                      json={"texts": [text]}, timeout=timeout)
    if r.status_code != 200:
        log.error(f"Analyze failed: {r.text}")
        sys.exit(1)
    data = r.json()
    if ctx.obj.get("json"):
        _output(ctx, data)
    else:
        info = data.get("data", data)
        for k, v in info.items():
            if k != "texts":
                log.info(f"  {k}: {v}")


@tokenizer.command("vocab", help="Show vocabulary")
@click.option("--limit", "-n", default=20, type=int, help="Max entries")
@click.pass_context
def tokenizer_vocab(ctx, limit):
    import requests
    timeout = ctx.obj.get("timeout", 10)
    r = requests.get(f"http://{ctx.obj['host']}:{ctx.obj['port']}/tokenizer/vocab?limit={limit}", timeout=timeout)
    if r.status_code != 200:
        log.error(f"Vocab failed: {r.text}")
        sys.exit(1)
    data = r.json()
    vocab = data.get("data", data).get("vocab", {})
    if ctx.obj.get("json"):
        _output(ctx, {"vocab": vocab})
    else:
        log.header("Vocabulary")
        for token_id, token_str in vocab.items():
            log.info(f"  {token_id}: {token_str}")


@tokenizer.command("merges", help="Show BPE merge rules")
@click.option("--limit", "-n", default=20, type=int, help="Max merges")
@click.pass_context
def tokenizer_merges(ctx, limit):
    import requests
    timeout = ctx.obj.get("timeout", 10)
    r = requests.get(f"http://{ctx.obj['host']}:{ctx.obj['port']}/tokenizer/merges?limit={limit}", timeout=timeout)
    if r.status_code != 200:
        log.error(f"Merges failed: {r.text}")
        sys.exit(1)
    data = r.json()
    merges = data.get("data", data).get("merges", [])
    if ctx.obj.get("json"):
        _output(ctx, {"merges": merges})
    else:
        log.header("BPE Merges")
        for m in merges:
            log.info(f"  {m}")


@tokenizer.command("train", help="Train tokenizer on texts")
@click.option("--vocab-size", type=int, default=512, help="Vocabulary size")
@click.option("--texts", default="", help="Comma-separated training texts")
@click.pass_context
def tokenizer_train(ctx, vocab_size, texts):
    import requests
    text_list = [t.strip() for t in texts.split(",") if t.strip()] if texts else []
    timeout = ctx.obj.get("timeout", 30)
    r = requests.post(f"http://{ctx.obj['host']}:{ctx.obj['port']}/tokenizer/train",
                      json={"vocab_size": vocab_size, "texts": text_list}, timeout=timeout)
    if r.status_code != 200:
        log.error(f"Train failed: {r.text}")
        sys.exit(1)
    log.success("Tokenizer trained")


@tokenizer.command("stats", help="Show tokenizer statistics")
@click.pass_context
def tokenizer_stats(ctx):
    import requests
    timeout = ctx.obj.get("timeout", 10)
    r = requests.get(f"http://{ctx.obj['host']}:{ctx.obj['port']}/tokenizer/stats", timeout=timeout)
    if r.status_code != 200:
        log.error(f"Stats failed: {r.text}")
        sys.exit(1)
    data = r.json()
    stats = data.get("data", data)
    if ctx.obj.get("json"):
        _output(ctx, stats)
    else:
        log.header("Tokenizer Stats")
        for k, v in stats.items():
            log.info(f"  {k}: {v}")


# ═══════════════════════════════════════════════════════════════════════
# vector  — init, upsert, search, stats
# ═══════════════════════════════════════════════════════════════════════


@cli.group(help="Vector store for semantic search")
def vector():
    pass


@vector.command("init", help="Initialize vector store")
@click.option("--provider", default="in_memory", help="Provider: in_memory, chromadb")
@click.option("--dimension", type=int, default=384, help="Embedding dimension")
@click.pass_context
def vector_init(ctx, provider, dimension):
    import requests
    timeout = ctx.obj.get("timeout", 10)
    r = requests.post(f"http://{ctx.obj['host']}:{ctx.obj['port']}/vector/init",
                      json={"provider": provider, "dimension": dimension}, timeout=timeout)
    if r.status_code != 200:
        log.error(f"Init failed: {r.text}")
        sys.exit(1)
    log.success(f"Vector store initialized: {provider} (dim={dimension})")


@vector.command("upsert", help="Insert or update vectors")
@click.argument("texts")
@click.option("--ids", default="", help="Comma-separated IDs")
@click.pass_context
def vector_upsert(ctx, texts, ids):
    import requests
    text_list = [t.strip() for t in texts.split(",") if t.strip()]
    id_list = [i.strip() for i in ids.split(",") if i.strip()] if ids else None
    timeout = ctx.obj.get("timeout", 10)
    r = requests.post(f"http://{ctx.obj['host']}:{ctx.obj['port']}/vector/upsert",
                      json={"texts": text_list, "ids": id_list}, timeout=timeout)
    if r.status_code != 200:
        log.error(f"Upsert failed: {r.text}")
        sys.exit(1)
    log.success(f"Upserted {len(text_list)} vectors")


@vector.command("search", help="Semantic search")
@click.argument("query")
@click.option("--top-k", type=int, default=5, help="Number of results")
@click.pass_context
def vector_search(ctx, query, top_k):
    import requests
    timeout = ctx.obj.get("timeout", 10)
    r = requests.post(f"http://{ctx.obj['host']}:{ctx.obj['port']}/vector/search",
                      json={"query": query, "top_k": top_k}, timeout=timeout)
    if r.status_code != 200:
        log.error(f"Search failed: {r.text}")
        sys.exit(1)
    data = r.json()
    results = data.get("data", data).get("results", [])
    if ctx.obj.get("json"):
        _output(ctx, {"results": results})
    else:
        log.header(f"Search: {query}")
        for i, res in enumerate(results):
            text = res.get("text", res.get("content", ""))[:80]
            score = res.get("score", 0)
            log.info(f"  {i+1}. [{score:.3f}] {text}")


@vector.command("stats", help="Show vector store stats")
@click.pass_context
def vector_stats(ctx):
    import requests
    timeout = ctx.obj.get("timeout", 10)
    r = requests.get(f"http://{ctx.obj['host']}:{ctx.obj['port']}/vector/stats", timeout=timeout)
    if r.status_code != 200:
        log.error(f"Stats failed: {r.text}")
        sys.exit(1)
    data = r.json()
    stats = data.get("data", data)
    if ctx.obj.get("json"):
        _output(ctx, stats)
    else:
        log.header("Vector Store Stats")
        for k, v in stats.items():
            log.info(f"  {k}: {v}")


# ═══════════════════════════════════════════════════════════════════════
# system  — status, info, health, stats, doctor, optimize, setup
# ═══════════════════════════════════════════════════════════════════════


@cli.group(help="System information, health, and environment tools")
def system():
    pass


@system.command("status", help="Show live system status")
@click.option("--watch", is_flag=True, help="Auto-refresh")
@click.option("--interval", default=3, type=int, help="Refresh interval")
@click.pass_context
def system_status(ctx, watch, interval):
    from commands.system import cmd_status
    cmd_status(_ns(watch=watch, interval=interval, json_output=ctx.obj.get("json"), quiet=ctx.obj.get("quiet"),
               timeout=ctx.obj.get("timeout", 10)))


@system.command("info", help="Show system information")
@click.pass_context
def system_info(ctx):
    from commands.system import cmd_system
    cmd_system(_ns(json_output=ctx.obj.get("json")))


@system.command("health", help="Quick API health check")
@click.pass_context
def system_health(ctx):
    from commands.dev import cmd_health
    args = _ns(host=ctx.obj["host"], port=ctx.obj["port"], json_output=ctx.obj.get("json"),
               timeout=ctx.obj.get("timeout", 10))
    cmd_health(args)


@system.command("stats", help="Show models/datasets statistics")
@click.pass_context
def system_stats(ctx):
    from commands.system import cmd_stats
    cmd_stats(_ns(json_output=ctx.obj.get("json")))


@system.command("doctor", help="Run environment checks")
@click.pass_context
def system_doctor(ctx):
    from commands.system import cmd_config_check
    cmd_config_check(_ns(json_output=ctx.obj.get("json")))


@system.command("config", help="Show or validate configuration")
@click.option("--validate", "do_validate", is_flag=True, help="Validate .env file")
@click.option("--env", default=".env", help="Dotenv file")
@click.option("--generate", "do_generate", is_flag=True, help="Generate secrets")
@click.option("--type", "secret_type", type=click.Choice(["api-key", "jwt-secret", "all"]), default="all")
def system_config(do_validate, env, do_generate, secret_type):
    if do_generate:
        from commands.system import cmd_config_generate
        cmd_config_generate(_ns(type=secret_type))
    elif do_validate:
        from commands.system import cmd_config_validate
        cmd_config_validate(_ns(env=env))
    else:
        from commands.system import cmd_config_check
        cmd_config_check(_ns())


@system.command("optimize", help="Show or apply optimization settings")
@click.option("--apply", "do_apply", is_flag=True, help="Apply optimizations")
def system_optimize(do_apply):
    from commands.system import cmd_optimize
    cmd_optimize(_ns(optimize=do_apply))


@system.command("setup", help="Bootstrap environment")
@click.option("--gpu", is_flag=True, help="GPU support")
@click.option("--docker-only", is_flag=True, help="Docker only")
@click.option("--local-only", is_flag=True, help="Local only")
@click.option("--venv", default=".venv", help="Virtual env directory")
def system_setup(gpu, docker_only, local_only, venv):
    from commands.system import cmd_setup
    args = _ns(gpu=gpu, docker_only=docker_only, local_only=local_only, venv=venv)
    cmd_setup(args)


@system.command("api", help="Test API endpoints or authentication")
@click.argument("action", type=click.Choice(["status", "test", "auth"]), default="status")
@click.pass_context
def system_api(ctx, action):
    from commands.dev import cmd_api_status, cmd_api_test, cmd_api_auth
    args = _ns(host=ctx.obj["host"], port=ctx.obj["port"])
    {
        "status": cmd_api_status,
        "test": cmd_api_test,
        "auth": cmd_api_auth,
    }[action](args)


# ═══════════════════════════════════════════════════════════════════════
# docker  — start, stop, status, logs, build, shell
# ═══════════════════════════════════════════════════════════════════════


@cli.group(help="Docker compose workflows")
def docker():
    pass


@docker.command("start", help="Start Docker services")
@click.option("--gpu", is_flag=True, help="Use GPU profile")
@click.option("--dev", is_flag=True, help="Use dev profile")
def docker_start(gpu, dev):
    _docker_action("start", _ns(gpu=gpu, dev=dev))


@docker.command("stop", help="Stop Docker services")
def docker_stop():
    _docker_action("stop", _ns())


@docker.command("status", help="Show Docker status")
def docker_status():
    _docker_action("status", _ns())


@docker.command("logs", help="Show Docker logs")
@click.argument("service", required=False)
def docker_logs(service):
    _docker_action("logs", _ns(service=service))


@docker.command("build", help="Build Docker images")
@click.option("--no-cache", is_flag=True, help="Build without cache")
def docker_build(no_cache):
    _docker_action("build", _ns(no_cache=no_cache))


@docker.command("shell", help="Shell into container")
@click.argument("service", default="api")
def docker_shell(service):
    _docker_action("shell", _ns(service=service))


# ═══════════════════════════════════════════════════════════════════════
# Simulate — boot kernel, load model, run inference, dump metrics
# ═══════════════════════════════════════════════════════════════════════


@cli.command(help="Boot kernel, load model, run inference — hardware simulator")
@click.option("--model", default="mock", help="Model name to load (default: mock)")
@click.option("--prompt", default="Hello, world", help="Prompt for generation")
@click.option("--max-tokens", default=20, type=int, help="Max tokens to generate")
@click.option("--iterations", default=1, type=int, help="Number of inference iterations")
@click.option("--layers", default=2, type=int, help="Number of transformer layers (mock model)")
@click.option("--d-model", default=64, type=int, help="Model dimension (mock model)")
@click.option("--vocab-size", default=256, type=int, help="Vocabulary size (mock model)")
@click.option("--profile", is_flag=True, help="Show detailed timing profile")
@click.option("--run-asm", "asm_source", default=None, help="Run VM assembly program instead of inference")
@click.option("--self-test", "do_self_test", is_flag=True, help="Run built-in VM self-test")
@click.pass_context
def simulate(ctx, model: str, prompt: str, max_tokens: int, iterations: int,
             layers: int, d_model: int, vocab_size: int, profile: bool,
             asm_source: str | None, do_self_test: bool):
    """Boot the kernel, load a model, run inference, and print metrics."""
    import time
    import sys
    import numpy as np

    # ANSI helpers
    _tty = sys.stdout.isatty()
    def _c(text, code):
        return f"{code}{text}\033[0m" if _tty else text
    _BOLD = "\033[1m"
    _DIM = "\033[2m"
    _CYAN = "\033[36m"
    _GREEN = "\033[32m"
    _YELLOW = "\033[33m"
    _MAGENTA = "\033[35m"
    _BLUE = "\033[34m"

    def _p(text=""):
        sys.stdout.write(text + "\n")
        sys.stdout.flush()

    def _table(headers, rows, col_styles=None):
        widths = [len(h) for h in headers]
        for row in rows:
            for i in range(min(len(row), len(widths))):
                widths[i] = max(widths[i], len(str(row[i])))
        hdr = "  ".join(_c(h.ljust(widths[i]), _BOLD, _tty) for i, h in enumerate(headers))
        sep = "  ".join("-" * w for w in widths)
        _p(hdr)
        _p(sep)
        for row in rows:
            cells = []
            for i in range(len(headers)):
                val = str(row[i]) if i < len(row) else ""
                cells.append(val.ljust(widths[i]))
            _p("  ".join(cells))

    _p(f"\n{_c('Kernel Simulation', _BOLD + _CYAN)}\n")

    # ── Self-test mode ──
    if do_self_test:
        from domains.shell.vm import self_test
        _p(f"{_c('Running VM self-test...', _BOLD)}\n")
        results = self_test()
        for line in results:
            _p(line)
        _p()
        return

    # ── Run assembly mode ──
    if asm_source:
        from domains.shell.vm import VMRunner
        _p(f"{_c('Running VM assembly...', _BOLD)}\n")
        runner = VMRunner()
        t0 = time.perf_counter()
        output = runner.assemble_and_run(asm_source, trace=profile)
        elapsed = time.perf_counter() - t0
        for line in output:
            _p(f"  {line}")
        _p(f"\n  {_c(f'Completed in {elapsed*1000:.2f}ms, {runner.cpu._step_count} steps', _DIM)}")
        if profile:
            trace = runner.cpu.get_trace()
            if trace:
                _p()
                _table(
                    ["Step", "PC", "Instruction", "Registers"],
                    [
                        [str(e.cycle), str(e.pc), e.instruction,
                         ", ".join(f"{k}={v}" for k, v in e.registers.items())]
                        for e in trace[:50]
                    ],
                )
                if len(trace) > 50:
                    _p(f"  ... ({len(trace)-50} more)")
        _p()
        return

    # ── Boot ──
    t0 = time.perf_counter()
    from domains.shell.kernel import Kernel
    k = Kernel()
    boot_msg = k.boot()
    t_boot = time.perf_counter() - t0
    _p(f"  {_c('ok', _GREEN)} Booted in {t_boot*1000:.1f}ms — {boot_msg}")

    try:
        # ── Register devices ──
        k.register_devices()
        _p(f"  {_c('ok', _GREEN)} {k.devices.stats()['total_devices']} devices registered")

        # ── Load model ──
        t1 = time.perf_counter()
        if model == "mock":
            class MockModel:
                def __init__(self):
                    self.call_count = 0
                    self.total_tokens = 0
                def __call__(self, input_ids):
                    self.call_count += 1
                    self.total_tokens += input_ids.size
                    return np.random.randn(input_ids.shape[0], input_ids.shape[1], vocab_size).astype(np.float32)
                def generate_numpy(self, prompt, max_tokens=10, temperature=1.0, **kw):
                    self.call_count += 1
                    self.total_tokens += max_tokens
                    return list(range(10, 10 + max_tokens))
                def forward(self, inputs):
                    self.call_count += 1
                    ids = inputs.get("input_ids", np.zeros((1, 10), dtype=np.int64))
                    self.total_tokens += ids.size
                    return {"logits": np.random.randn(ids.shape[0], ids.shape[1], vocab_size).astype(np.float32)}
            mock = MockModel()
            k.engine.load_model(model, mock)
        else:
            from domains.shell.kernel_npu import NPUDevice
            npu = NPUDevice(name="npu")
            npu.open()
            result = npu.load_model(model, f"huggingface:{model}")
            if not result.success:
                _p(f"  {_c(f'⚠ Could not load \'{model}\': {result.error}', _YELLOW)}")
                _p(f"  {_c('Falling back to mock model. Install transformers for real models.', _DIM)}")
                class FallbackModel:
                    def __init__(self):
                        self.call_count = 0
                        self.total_tokens = 0
                    def __call__(self, input_ids):
                        self.call_count += 1
                        self.total_tokens += input_ids.size
                        return np.random.randn(input_ids.shape[0], input_ids.shape[1], vocab_size).astype(np.float32)
                    def generate_numpy(self, prompt, max_tokens=10, temperature=1.0, **kw):
                        self.call_count += 1
                        self.total_tokens += max_tokens
                        return list(range(10, 10 + max_tokens))
                    def forward(self, inputs):
                        self.call_count += 1
                        ids = inputs.get("input_ids", np.zeros((1, 10), dtype=np.int64))
                        self.total_tokens += ids.size
                        return {"logits": np.random.randn(ids.shape[0], ids.shape[1], vocab_size).astype(np.float32)}
                k.engine.load_model(model, FallbackModel())
            else:
                provider = npu._models[model].provider
                k.engine.load_model(model, provider)
        t_load = time.perf_counter() - t1
        _p(f"  {_c('ok', _GREEN)} Model '{model}' loaded in {t_load*1000:.1f}ms")

        # ── Tokenize ──
        t2 = time.perf_counter()
        tokens = k.tokenize(prompt)
        t_tok = time.perf_counter() - t2
        _p(f"  {_c('ok', _GREEN)} Tokenized '{prompt[:40]}...' -> {len(tokens)} tokens in {t_tok*1000:.2f}ms")

        # ── Create inference process ──
        from domains.shell.kernel_neural import NeuralProcessType
        proc = k.create_neural_process("sim-infer", NeuralProcessType.INFERENCE, model_name=model)

        # ── Warmup ──
        input_ids = np.array([tokens])
        _ = k.forward(proc, {"input_ids": input_ids})

        # ── Inference iterations ──
        latencies = []
        tokens_generated = []
        for i in range(iterations):
            t3 = time.perf_counter()
            gen = k.generate(model, prompt, max_tokens=max_tokens)
            t_inf = time.perf_counter() - t3
            latencies.append(t_inf)
            tokens_generated.append(gen["token_count"] if gen else 0)

        # ── Metrics ──
        avg_latency = sum(latencies) / len(latencies)
        total_tokens = sum(tokens_generated)
        throughput = total_tokens / sum(latencies) if sum(latencies) > 0 else 0

        # ── KV Cache ──
        cache = k.create_kv_cache("sim-cache", num_layers=layers, head_dim=d_model // 4)
        cache.initialize(num_heads=4)
        for step in range(min(10, max_tokens)):
            k0 = np.random.randn(4, d_model // 4)
            v0 = np.random.randn(4, d_model // 4)
            cache.update(step % layers, k0, v0)
            cache.advance(1)

        # ── Neural stats ──
        ns = k.neural_stats()

        # ── Kernel stats ──
        ks = k.stats()

        # ── Print results ──
        _p()
        _table(
            ["Metric", "Value"],
            [
                ["Boot time", f"{t_boot*1000:.1f}ms"],
                ["Model load", f"{t_load*1000:.1f}ms"],
                ["Tokenize", f"{t_tok*1000:.2f}ms"],
                ["Tokens in prompt", str(len(tokens))],
                ["Iterations", str(iterations)],
                ["Avg latency", f"{avg_latency*1000:.1f}ms"],
                ["Total tokens generated", str(total_tokens)],
                ["Throughput", f"{throughput:.1f} tok/s"],
                ["Processes", str(ks["process_count"])],
                ["KV cache layers", str(ns["kv_caches"])],
                ["KV cache memory", f"{ns['gradient_accumulator']['step_count']} steps"],
                ["Uptime", f"{k.uptime:.2f}s"],
            ],
        )

        if profile:
            _p()
            _table(
                ["Iter", "Latency", "Tokens", "tok/s"],
                [
                    [str(i + 1), f"{lat*1000:.1f}ms", str(tok),
                     f"{tok / lat:.1f}" if lat > 0 else "0.0"]
                    for i, (lat, tok) in enumerate(zip(latencies, tokens_generated))
                ],
            )

        _p(f"\n{_c('Simulation complete.', _BOLD + _GREEN)}\n")

    finally:
        k.shutdown()
        _p(f"  {_c('Kernel shut down.', _DIM)}")


# ═══════════════════════════════════════════════════════════════════════
# Collections — data feed ingestion
# ═══════════════════════════════════════════════════════════════════════


@cli.group(help="Collect data from files, URLs, RSS feeds, and APIs")
def collect():
    pass


@collect.command("file", help="Collect data from a local file")
@click.argument("path")
@click.option("--output", "-o", default=None, help="Output JSONL file")
@click.option("--min-length", default=10, type=int, help="Min record length")
@click.option("--dedup/--no-dedup", default=True, help="Deduplicate records")
def collect_file(path, output, min_length, dedup):
    from domains.collections import FileSource, MemoryStore, FileStore, Collector
    from domains.collections import LengthFilter, DedupFilter
    source = FileSource(path)
    store = FileStore(output) if output else MemoryStore()
    filters = []
    if min_length > 0:
        filters.append(LengthFilter(min_length=min_length))
    if dedup:
        filters.append(DedupFilter())
    collector = Collector(source, store, filters=filters)
    count = collector.collect()
    log.success(f"Collected {count} records from {path}")
    if output:
        log.info(f"Output: {output}")


@collect.command("url", help="Collect data from a URL")
@click.argument("url")
@click.option("--output", "-o", default=None, help="Output JSONL file")
@click.option("--min-length", default=10, type=int, help="Min record length")
def collect_url(url, output, min_length):
    from domains.collections import UrlSource, MemoryStore, FileStore, Collector
    from domains.collections import LengthFilter
    source = UrlSource(url)
    store = FileStore(output) if output else MemoryStore()
    filters = [LengthFilter(min_length=min_length)] if min_length > 0 else []
    collector = Collector(source, store, filters=filters)
    count = collector.collect()
    log.success(f"Collected {count} records from {url}")
    if output:
        log.info(f"Output: {output}")


@collect.command("rss", help="Collect data from an RSS/Atom feed")
@click.argument("url")
@click.option("--output", "-o", default=None, help="Output JSONL file")
def collect_rss(url, output):
    from domains.collections import RssSource, MemoryStore, FileStore, Collector
    source = RssSource(url)
    store = FileStore(output) if output else MemoryStore()
    collector = Collector(source, store)
    count = collector.collect()
    log.success(f"Collected {count} records from RSS feed")
    if output:
        log.info(f"Output: {output}")


@collect.command("merge", help="Merge multiple JSONL files into one")
@click.argument("inputs", nargs=-1, required=True)
@click.option("--output", "-o", required=True, help="Output JSONL file")
def collect_merge(inputs, output):
    import json
    from pathlib import Path
    count = 0
    with open(output, "w") as out_f:
        for input_path in inputs:
            p = Path(input_path)
            if not p.exists():
                log.warning(f"Skipping {input_path} (not found)")
                continue
            with open(p) as in_f:
                for line in in_f:
                    line = line.strip()
                    if line:
                        out_f.write(line + "\n")
                        count += 1
    log.success(f"Merged {len(inputs)} files -> {output} ({count} records)")


@collect.command("stats", help="Show collection statistics")
@click.argument("path")
def collect_stats(path):
    import json
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        log.error(f"File not found: {path}")
        return
    count = 0
    total_bytes = 0
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                count += 1
                total_bytes += len(line)
    log.header(f"Collection Stats: {p.name}")
    log.key_value("Records", str(count))
    log.key_value("Total Size", f"{total_bytes:,} bytes")
    log.key_value("Avg Size", f"{total_bytes // max(count, 1):,} bytes")


# ═══════════════════════════════════════════════════════════════════════
# companion  — status, chat, personality, preset
# ═══════════════════════════════════════════════════════════════════════


@cli.group(help="AI companion management and chat")
def companion():
    pass


@companion.command("status", help="Show companion status")
@click.pass_context
def companion_status(ctx):
    import requests
    timeout = ctx.obj.get("timeout", 10)
    r = requests.get(f"http://{ctx.obj['host']}:{ctx.obj['port']}/companion/status", timeout=timeout)
    if r.status_code != 200:
        log.error(f"Status failed: {r.text}")
        sys.exit(1)
    data = r.json()
    stats = data.get("data", data)
    if ctx.obj.get("json"):
        _output(ctx, stats)
    else:
        log.header("Companion Status")
        for k, v in stats.items():
            log.info(f"  {k}: {v}")


@companion.command("chat", help="Chat with companion")
@click.argument("message")
@click.option("--user-name", default="", help="Your name")
@click.option("--mood", default="", help="Your current mood")
@click.pass_context
def companion_chat(ctx, message, user_name, mood):
    import requests
    timeout = ctx.obj.get("timeout", 30)
    payload = {"message": message}
    if user_name:
        payload["user_name"] = user_name
    if mood:
        payload["user_mood"] = mood
    r = requests.post(f"http://{ctx.obj['host']}:{ctx.obj['port']}/companion/chat",
                      json=payload, timeout=timeout)
    if r.status_code != 200:
        log.error(f"Chat failed: {r.text}")
        sys.exit(1)
    data = r.json()
    if ctx.obj.get("json"):
        _output(ctx, data)
    else:
        resp = data.get("data", data).get("response", str(data))
        log.info(resp)


@companion.command("personality", help="Show companion personality")
@click.pass_context
def companion_personality(ctx):
    import requests
    timeout = ctx.obj.get("timeout", 10)
    r = requests.get(f"http://{ctx.obj['host']}:{ctx.obj['port']}/companion/personality", timeout=timeout)
    if r.status_code != 200:
        log.error(f"Failed: {r.text}")
        sys.exit(1)
    data = r.json()
    if ctx.obj.get("json"):
        _output(ctx, data)
    else:
        p = data.get("data", data)
        log.header("Companion Personality")
        for k, v in p.items():
            log.info(f"  {k}: {v}")


@companion.command("preset", help="Use a preset personality")
@click.argument("name")
@click.pass_context
def companion_preset(ctx, name):
    import requests
    timeout = ctx.obj.get("timeout", 10)
    r = requests.post(f"http://{ctx.obj['host']}:{ctx.obj['port']}/companion/preset",
                      json={"preset": name}, timeout=timeout)
    if r.status_code != 200:
        log.error(f"Preset failed: {r.text}")
        sys.exit(1)
    log.success(f"Applied preset: {name}")


# ═══════════════════════════════════════════════════════════════════════
# images  — generate, gallery, styles
# ═══════════════════════════════════════════════════════════════════════


@cli.group(help="Image generation and gallery")
def images():
    pass


@images.command("generate", help="Generate an image from text")
@click.argument("prompt")
@click.option("--style", type=click.Choice(["realistic", "cartoon", "watercolor", "sketch", "fantasy"]),
              default="realistic", help="Image style")
@click.option("--output", "-o", help="Save to file path")
@click.pass_context
def images_generate(ctx, prompt, style, output):
    import requests
    timeout = ctx.obj.get("timeout", 60)
    r = requests.post(f"http://{ctx.obj['host']}:{ctx.obj['port']}/images/generate",
                      json={"prompt": prompt, "style": style}, timeout=timeout)
    if r.status_code != 200:
        log.error(f"Generate failed: {r.text}")
        sys.exit(1)
    data = r.json()
    if ctx.obj.get("json"):
        _output(ctx, data)
    else:
        img_data = data.get("data", data)
        img_id = img_data.get("id", "?")
        log.success(f"Generated image: {img_id} (style={style})")
        if output:
            import base64
            b64 = img_data.get("image", "")
            if b64 and "," in b64:
                b64 = b64.split(",", 1)[1]
            with open(output, "wb") as f:
                f.write(base64.b64decode(b64))
            log.info(f"Saved to: {output}")


@images.command("gallery", help="List generated images")
@click.option("--limit", "-n", default=10, type=int)
@click.pass_context
def images_gallery(ctx, limit):
    import requests
    timeout = ctx.obj.get("timeout", 10)
    r = requests.get(f"http://{ctx.obj['host']}:{ctx.obj['port']}/images/gallery?limit={limit}", timeout=timeout)
    if r.status_code != 200:
        log.error(f"Gallery failed: {r.text}")
        sys.exit(1)
    data = r.json()
    images_list = data.get("data", data).get("images", [])
    if ctx.obj.get("json"):
        _output(ctx, {"images": images_list})
    else:
        log.header("Image Gallery")
        for img in images_list:
            log.info(f"  {img.get('id', '?')} — {img.get('prompt', '')[:60]}")


@images.command("styles", help="List available styles")
@click.pass_context
def images_styles(ctx):
    import requests
    timeout = ctx.obj.get("timeout", 10)
    r = requests.get(f"http://{ctx.obj['host']}:{ctx.obj['port']}/images/styles", timeout=timeout)
    if r.status_code != 200:
        log.error(f"Styles failed: {r.text}")
        sys.exit(1)
    data = r.json()
    styles = data.get("data", data).get("styles", [])
    if ctx.obj.get("json"):
        _output(ctx, {"styles": styles})
    else:
        log.header("Available Styles")
        for s in styles:
            log.info(f"  {s}")


# ═══════════════════════════════════════════════════════════════════════
# multimodal  — status, vision, speech, dpo, video
# ═══════════════════════════════════════════════════════════════════════


@cli.group(help="Multimodal capabilities (vision, speech, video)")
def multimodal():
    pass


@multimodal.command("status", help="Show multimodal engine status")
@click.pass_context
def multimodal_status(ctx):
    import requests
    timeout = ctx.obj.get("timeout", 10)
    r = requests.get(f"http://{ctx.obj['host']}:{ctx.obj['port']}/multimodal/status", timeout=timeout)
    if r.status_code != 200:
        log.error(f"Status failed: {r.text}")
        sys.exit(1)
    data = r.json()
    stats = data.get("data", data)
    if ctx.obj.get("json"):
        _output(ctx, stats)
    else:
        log.header("Multimodal Status")
        for k, v in stats.items():
            log.info(f"  {k}: {v}")


@multimodal.command("dpo", help="Trigger DPO training")
@click.option("--max-pairs", type=int, default=6, help="Max preference pairs")
@click.option("--lr", type=float, default=5e-6, help="Learning rate")
@click.pass_context
def multimodal_dpo(ctx, max_pairs, lr):
    import requests
    timeout = ctx.obj.get("timeout", 60)
    r = requests.post(f"http://{ctx.obj['host']}:{ctx.obj['port']}/multimodal/dpo/trigger",
                      json={"max_pairs": max_pairs, "learning_rate": lr}, timeout=timeout)
    if r.status_code != 200:
        log.error(f"DPO failed: {r.text}")
        sys.exit(1)
    log.success("DPO training triggered")


@multimodal.command("video-train", help="Train video model")
@click.argument("data_path")
@click.option("--epochs", type=int, default=5)
@click.option("--batch-size", type=int, default=2)
@click.pass_context
def multimodal_video_train(ctx, data_path, epochs, batch_size):
    import requests
    timeout = ctx.obj.get("timeout", 120)
    r = requests.post(f"http://{ctx.obj['host']}:{ctx.obj['port']}/multimodal/video/train",
                      json={"data_path": data_path, "epochs": epochs, "batch_size": batch_size},
                      timeout=timeout)
    if r.status_code != 200:
        log.error(f"Video train failed: {r.text}")
        sys.exit(1)
    log.success("Video training started")


# ═══════════════════════════════════════════════════════════════════════
# meta-weights  — get, stats
# ═══════════════════════════════════════════════════════════════════════


@cli.group(help="Feedback-driven meta-weight adaptation")
def meta_weights():
    pass


@meta_weights.command("get", help="Get meta-weight adjustments")
@click.argument("message")
@click.option("--k", type=int, default=5, help="Number of similar samples")
@click.pass_context
def meta_weights_get(ctx, message, k):
    import requests
    timeout = ctx.obj.get("timeout", 10)
    r = requests.post(f"http://{ctx.obj['host']}:{ctx.obj['port']}/meta-weights/get",
                      json={"user_message": message, "k": k}, timeout=timeout)
    if r.status_code != 200:
        log.error(f"Failed: {r.text}")
        sys.exit(1)
    data = r.json()
    if ctx.obj.get("json"):
        _output(ctx, data)
    else:
        w = data.get("data", data)
        log.header("Meta-Weights")
        for k, v in w.items():
            log.info(f"  {k}: {v}")


@meta_weights.command("stats", help="Show meta-weight statistics")
@click.pass_context
def meta_weights_stats(ctx):
    import requests
    timeout = ctx.obj.get("timeout", 10)
    r = requests.get(f"http://{ctx.obj['host']}:{ctx.obj['port']}/meta-weights/stats", timeout=timeout)
    if r.status_code != 200:
        log.error(f"Stats failed: {r.text}")
        sys.exit(1)
    data = r.json()
    stats = data.get("data", data)
    if ctx.obj.get("json"):
        _output(ctx, stats)
    else:
        log.header("Meta-Weight Stats")
        for k, v in stats.items():
            log.info(f"  {k}: {v}")


# ═══════════════════════════════════════════════════════════════════════
# learn  — search, feed, status, train, knowledge
# ═══════════════════════════════════════════════════════════════════════


@cli.group(help="Continual learning from web, feeds, and knowledge")
def learn():
    pass


@learn.command("search", help="Search web and learn from results")
@click.argument("query")
@click.option("--max-results", type=int, default=5, help="Max results")
@click.pass_context
def learn_search(ctx, query, max_results):
    import requests
    timeout = ctx.obj.get("timeout", 60)
    r = requests.post(f"http://{ctx.obj['host']}:{ctx.obj['port']}/learn/search",
                      json={"query": query, "max_results": max_results}, timeout=timeout)
    if r.status_code != 200:
        log.error(f"Search failed: {r.text}")
        sys.exit(1)
    data = r.json()
    if ctx.obj.get("json"):
        _output(ctx, data)
    else:
        info = data.get("data", data)
        log.info(f"Tokens ingested: {info.get('tokens_ingested', 0)}")
        log.info(f"New facts: {info.get('new_facts', 0)}")
        log.info(f"Rejected: {info.get('rejected', 0)}")


@learn.command("status", help="Show learner status")
@click.pass_context
def learn_status(ctx):
    import requests
    timeout = ctx.obj.get("timeout", 10)
    r = requests.get(f"http://{ctx.obj['host']}:{ctx.obj['port']}/learn/status", timeout=timeout)
    if r.status_code != 200:
        log.error(f"Status failed: {r.text}")
        sys.exit(1)
    data = r.json()
    stats = data.get("data", data)
    if ctx.obj.get("json"):
        _output(ctx, stats)
    else:
        log.header("Learner Status")
        for k, v in stats.items():
            log.info(f"  {k}: {v}")


@learn.command("knowledge", help="Query learned knowledge")
@click.argument("query", required=False)
@click.option("--topic", default="", help="Filter by topic")
@click.pass_context
def learn_knowledge(ctx, query, topic):
    import requests
    timeout = ctx.obj.get("timeout", 10)
    params = {}
    if query:
        params["q"] = query
    if topic:
        params["topic"] = topic
    r = requests.get(f"http://{ctx.obj['host']}:{ctx.obj['port']}/learn/knowledge",
                     params=params, timeout=timeout)
    if r.status_code != 200:
        log.error(f"Knowledge query failed: {r.text}")
        sys.exit(1)
    data = r.json()
    facts = data.get("data", data).get("facts", [])
    if ctx.obj.get("json"):
        _output(ctx, {"facts": facts})
    else:
        log.header("Learned Knowledge")
        for f in facts[:20]:
            topic = f.get("topic", "?")
            text = f.get("text", "")[:80]
            log.info(f"  [{topic}] {text}")


@learn.command("train", help="Force a training step")
@click.pass_context
def learn_train(ctx):
    import requests
    timeout = ctx.obj.get("timeout", 60)
    r = requests.post(f"http://{ctx.obj['host']}:{ctx.obj['port']}/learn/train", timeout=timeout)
    if r.status_code != 200:
        log.error(f"Train failed: {r.text}")
        sys.exit(1)
    log.success("Training step completed")


@learn.command("ingest", help="Ingest raw text")
@click.argument("text")
@click.pass_context
def learn_ingest(ctx, text):
    import requests
    timeout = ctx.obj.get("timeout", 30)
    r = requests.post(f"http://{ctx.obj['host']}:{ctx.obj['port']}/learn/ingest",
                      json={"text": text}, timeout=timeout)
    if r.status_code != 200:
        log.error(f"Ingest failed: {r.text}")
        sys.exit(1)
    log.success("Text ingested")


# ═══════════════════════════════════════════════════════════════════════
# World rendering
# ═══════════════════════════════════════════════════════════════════════


@cli.group(help="Render and simulate the programmable world")
def world():
    pass


@world.command("render", help="Render the current world state")
@click.option("--width", default=160, type=int, help="Render width")
@click.option("--height", default=120, type=int, help="Render height")
@click.option("--samples", default=16, type=int, help="Render samples")
@click.option("--output", "-o", default=None, help="Output file (PPM)")
@click.option("--neural", is_flag=True, help="Run neural processing on render")
def world_render(width, height, samples, output, neural):
    from domains.shell.world_render import RenderBridge, NeuralRenderBridge, RenderConfig
    from domains.shell.simulation import WorldGrid
    import numpy as np

    cfg = RenderConfig(width=width, height=height, samples=samples)

    if neural:
        bridge = NeuralRenderBridge(cfg)
    else:
        bridge = RenderBridge(cfg)

    world = WorldGrid()
    for x in range(10, 54):
        world.material[world.idx(x, 0, 32)] = 1
    for x in range(20, 44):
        world.material[world.idx(x, 0, 32)] = 2
        world.energy[world.idx(x, 0, 32)] = 3.0
    world.material[world.idx(32, 1, 32)] = 4
    world.energy[world.idx(32, 1, 32)] = 5.0

    log.header("Rendering world...")
    bridge.build_scene(world)
    image = bridge.render()
    log.success(f"Rendered {image.shape[1]}x{image.shape[0]} image ({bridge.stats['total_time_ms']:.0f}ms)")

    if output:
        img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        header = f"P6\n{img_uint8.shape[1]} {img_uint8.shape[0]}\n255\n"
        with open(output, "wb") as f:
            f.write(header.encode() + img_uint8.tobytes())
        log.info(f"Saved: {output}")

    if neural:
        result = bridge.process_neural()
        emb = result.get("embedding")
        if emb is not None:
            log.key_value("Embedding dim", str(len(emb)))
            log.key_value("Embedding norm", f"{np.linalg.norm(emb):.4f}")
        desc = bridge.get_descriptor()
        log.key_value("Scene features", str(len(desc.get("tensor_stats", {}))))


@world.command("tick", help="Run simulation ticks with optional rendering")
@click.option("--ticks", default=5, type=int, help="Number of ticks")
@click.option("--babies", default=4, type=int, help="Number of baby agents")
@click.option("--render", is_flag=True, help="Enable rendering")
@click.option("--neural", is_flag=True, help="Enable neural processing")
@click.option("--verbose", is_flag=True, help="Verbose output")
def world_tick(ticks, babies, render, neural, verbose):
    from domains.shell.simulation import SimScene, Simulation, WorldParams
    from domains.shell.world_render import RenderBridge, NeuralRenderBridge, RenderConfig

    params = WorldParams()
    scene = SimScene(params)

    for _ in range(babies):
        from domains.shell.simulation import SimBaby, Entity, EntityType
        baby = SimBaby()
        baby.entity.position[0] = 32 + np.random.randint(-10, 10)
        baby.entity.position[2] = 32 + np.random.randint(-10, 10)
        scene.add_baby(baby)

    render_bridge = None
    if render or neural:
        if neural:
            render_bridge = NeuralRenderBridge()
        else:
            render_bridge = RenderBridge()

    sim = Simulation(scene, max_ticks=ticks, verbose=verbose, render_bridge=render_bridge)
    log.header(f"Running {ticks} ticks with {len(scene.babies)} babies...")
    sim.run()
    summary = sim.summary()

    log.key_value("Ticks", str(summary.get("total_ticks", 0)))
    log.key_value("Babies at end", str(summary.get("alive_at_end", False)))
    log.key_value("Avg energy", f"{summary.get('avg_energy', 0):.1f}")
    log.key_value("Cells written", str(summary.get("total_cells_written", 0)))

    if render_bridge:
        log.key_value("Renders", str(render_bridge.stats.get("renders", 0)))
        log.key_value("Render time", f"{render_bridge.stats.get('total_time_ms', 0):.0f}ms")


@world.command("analyze", help="Analyze render history over simulation ticks")
@click.option("--ticks", default=20, type=int, help="Number of ticks to simulate")
@click.option("--babies", default=4, type=int, help="Number of baby agents")
@click.option("--threshold", default=0.1, type=float, help="Change detection threshold")
def world_analyze(ticks, babies, threshold):
    from domains.shell.simulation import SimScene, Simulation, WorldParams
    from domains.shell.world_render import RenderBridge, RenderAnalyzer, RenderConfig

    config = RenderConfig(width=64, height=48, samples=1)
    bridge = RenderBridge(config)

    params = WorldParams()
    scene = SimScene(params)

    for _ in range(babies):
        from domains.shell.simulation import SimBaby
        baby = SimBaby()
        baby.entity.position[0] = 32 + np.random.randint(-10, 10)
        baby.entity.position[2] = 32 + np.random.randint(-10, 10)
        scene.add_baby(baby)

    sim = Simulation(scene, max_ticks=ticks, render_bridge=bridge)
    analyzer = RenderAnalyzer(bridge._history if hasattr(bridge, '_history') else None)

    sim.run()

    for i, entry in enumerate(bridge._history._entries if hasattr(bridge, '_history') else []):
        analyzer.history.add(entry["image"], tick=entry["tick"])

    summary = analyzer.summary()
    log.header("Render Analysis")
    log.key_value("Total renders", str(summary.get("count", 0)))
    log.key_value("Significant changes", str(summary.get("significant_changes", 0)))
    if summary.get("mean_range"):
        log.key_value("Mean range", f"{summary['mean_range'][0]:.4f} - {summary['mean_range'][1]:.4f}")
    if summary.get("mean_trend") is not None:
        log.key_value("Mean trend", f"{summary['mean_trend']:+.4f}")

    changes = analyzer.detect_significant_changes(threshold)
    if changes:
        log.header("Significant Changes")
        for c in changes:
            log.info(f"  Tick {c['tick_from']} -> {c['tick_to']}: "
                     f"{c['change_ratio']:.1%} changed, MSE={c['mse']:.6f}")


@world.command("diff", help="Compare two render images")
@click.argument("image_a", type=click.Path(exists=True))
@click.argument("image_b", type=click.Path(exists=True))
def world_diff(image_a, image_b):
    from domains.shell.world_render import RenderDiff
    from PIL import Image as PILImage

    a = np.array(PILImage.open(image_a)).astype(np.float32) / 255.0
    b = np.array(PILImage.open(image_b)).astype(np.float32) / 255.0

    diff = RenderDiff(a, b)
    s = diff.summary()

    log.header("Render Diff")
    log.key_value("MSE", f"{s['mse']:.6f}")
    log.key_value("MAE", f"{s['mae']:.6f}")
    log.key_value("Max diff", f"{s['max_diff']:.4f}")
    log.key_value("Changed pixels", f"{s['changed_pixels']}/{s['total_pixels']} ({s['change_ratio']:.1%})")
    log.key_value("Mean A", f"{s['mean_a']:.4f}")
    log.key_value("Mean B", f"{s['mean_b']:.4f}")


@world.command("ingest", help="Feed data into the world grid")
@click.argument("source_type", type=click.Choice(["file", "url", "rss", "records"]))
@click.argument("source_value")
@click.option("--radius", default=15, type=int, help="Placement radius around center")
@click.option("--decay", default=0.95, type=float, help="Energy decay rate per tick")
@click.option("--verbose", is_flag=True, help="Verbose output")
def world_ingest(source_type, source_value, radius, decay, verbose):
    from domains.collections.perception import WorldPerception, PerceptionConfig
    from domains.collections.sources import FileSource, UrlSource, RssSource, GeneratorSource, Record
    from domains.shell.simulation import WorldGrid

    config = PerceptionConfig(radius=radius, decay_rate=decay)
    perception = WorldPerception(config)
    world = WorldGrid()

    if source_type == "file":
        records = []
        with open(source_value, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(Record(content=line))
        events = perception.ingest_records(records)
    elif source_type == "url":
        from domains.collections.sources import UrlSource
        source = UrlSource(source_value)
        events = perception.ingest_source(source)
    elif source_type == "rss":
        from domains.collections.sources import RssSource
        source = RssSource(source_value)
        events = perception.ingest_source(source)
    else:
        records = [Record(content=source_value)]
        events = perception.ingest_records(records)

    perception.apply_to_grid(world, events)

    log.header("World Ingestion")
    log.key_value("Records ingested", str(len(events)))
    log.key_value("Grid cells filled", str(np.sum(world.material != 0)))
    log.key_value("Avg energy", f"{np.mean(world.energy[world.material != 0]):.2f}" if np.any(world.material != 0) else "0.00")

    if verbose:
        summary = perception.summary()
        for cls, count in summary.get("material_counts", {}).items():
            log.key_value(f"  {cls}", str(count))


# ═══════════════════════════════════════════════════════════════════════
# vm  — Virtual Machine
# ═══════════════════════════════════════════════════════════════════════


@cli.group(help="x86 Virtual Machine console and management")
def vm():
    pass


@vm.command("status", help="Show VM status and information")
def vm_status():
    from commands.vm import cmd_vm
    cmd_vm(_ns())


@vm.command("run", help="Run assembly code in the VM")
@click.argument("source", required=False)
@click.option("--file", "-f", help="File containing assembly source")
def vm_run(source, file):
    from commands.vm import cmd_vm_run
    cmd_vm_run(_ns(source=source, file=file))


@vm.command("list", help="List available VM programs")
def vm_list():
    from commands.vm import cmd_vm_list
    cmd_vm_list(_ns())


@vm.command("info", help="Show detailed VM information")
def vm_info_cmd():
    from commands.vm import cmd_vm_info
    cmd_vm_info(_ns())


@vm.command("debug", help="Debug assembly code interactively or from script")
@click.argument("source", required=False)
@click.option("--file", "-f", help="File containing assembly source")
@click.option("--script", "-s", help="Script file with debug commands (non-interactive)")
def vm_debug(source, file, script):
    from commands.vm import cmd_vm_debug
    cmd_vm_debug(_ns(source=source, file=file, script=script))


# ═══════════════════════════════════════════════════════════════════════
# build  — Buildroot image building
# ═══════════════════════════════════════════════════════════════════════


@cli.group(help="Buildroot image building for v86 browser VM")
def build():
    pass


@build.command("run", help="Build a Buildroot image")
def build_run():
    from commands.build import cmd_build
    cmd_build(_ns())


@build.command("init", help="Initialize Buildroot build environment")
@click.option("--clean", is_flag=True, help="Clean first, then set up")
def build_init(clean):
    from commands.build import cmd_build_init
    cmd_build_init(_ns(clean=clean))


@build.command("clean", help="Clean build output")
def build_clean():
    from commands.build import cmd_build_clean
    cmd_build_clean(_ns())


@build.command("status", help="Show build status")
def build_status():
    from commands.build import cmd_build_status
    cmd_build_status(_ns())


@build.command("install", help="Install image to web public directory")
def build_install():
    from commands.build import cmd_build_install
    cmd_build_install(_ns())


# ═══════════════════════════════════════════════════════════════════════
# voice  — text-to-speech and speech-to-text
# ═══════════════════════════════════════════════════════════════════════


@cli.group(help="Text-to-speech and speech-to-text via the API")
def voice():
    pass


@voice.command("tts", help="Convert text to speech audio")
@click.argument("text")
@click.option("--output", "-o", help="Output file path (default: tts_output.wav)")
@click.option("--play", is_flag=True, help="Play audio after generating")
def voice_tts(text, output, play):
    from commands.voice import cmd_voice_tts
    cmd_voice_tts(_ns(text=text, output=output, play=play))


@voice.command("stt", help="Transcribe an audio file to text")
@click.argument("file")
@click.option("--language", "-l", default="en", help="Audio language (default: en)")
@click.option("--verbose", "-v", is_flag=True, help="Show confidence and language info")
def voice_stt(file, language, verbose):
    from commands.voice import cmd_voice_stt
    cmd_voice_stt(_ns(file=file, language=language, verbose=verbose))


# ═══════════════════════════════════════════════════════════════════════
# security  — audit, keys
# ═══════════════════════════════════════════════════════════════════════


@cli.group(help="Security audit logs and API key management")
def security():
    pass


@security.command("audit", help="Show audit logs")
@click.option("--limit", "-n", default=20, type=int, help="Max entries")
@click.option("--type", "event_type", default="", help="Filter by event type")
@click.option("--history", is_flag=True, help="Read from persisted audit.log")
@click.pass_context
def security_audit(ctx, limit, event_type, history):
    import requests
    timeout = ctx.obj.get("timeout", 10)
    params = {"limit": limit}
    if event_type:
        params["event_type"] = event_type
    if history:
        params["history"] = True
    r = requests.get(f"http://{ctx.obj['host']}:{ctx.obj['port']}/security/audit",
                     params=params, timeout=timeout)
    if r.status_code != 200:
        log.error(f"Audit failed: {r.text}")
        sys.exit(1)
    data = r.json()
    logs = data.get("data", data).get("logs", [])
    if ctx.obj.get("json"):
        _output(ctx, {"logs": logs})
    else:
        log.header(f"Audit Logs ({len(logs)} entries)")
        for entry in logs:
            event = entry.get("event_type", "?")
            ts = entry.get("timestamp", "")[:19]
            detail = entry.get("detail", "")[:60]
            log.info(f"  [{ts}] {event} — {detail}")


@security.command("keys", help="Show API key info")
@click.pass_context
def security_keys(ctx):
    import requests
    timeout = ctx.obj.get("timeout", 10)
    r = requests.get(f"http://{ctx.obj['host']}:{ctx.obj['port']}/security/keys", timeout=timeout)
    if r.status_code != 200:
        log.error(f"Keys failed: {r.text}")
        sys.exit(1)
    data = r.json()
    info = data.get("data", data)
    if ctx.obj.get("json"):
        _output(ctx, info)
    else:
        count = info.get("count", 0)
        configured = info.get("configured", False)
        log.info(f"API keys configured: {configured} ({count} keys)")


# ═══════════════════════════════════════════════════════════════════════
# docstore  — list, get, put, delete, collections
# ═══════════════════════════════════════════════════════════════════════


@cli.group(help="Server-side document store (browser chat DB)")
def docstore():
    pass


@docstore.command("collections", help="List document collections")
@click.pass_context
def docstore_collections(ctx):
    import requests
    timeout = ctx.obj.get("timeout", 10)
    r = requests.get(f"http://{ctx.obj['host']}:{ctx.obj['port']}/docstore/collections", timeout=timeout)
    if r.status_code != 200:
        log.error(f"Failed: {r.text}")
        sys.exit(1)
    data = r.json()
    cols = data.get("data", data).get("collections", [])
    if ctx.obj.get("json"):
        _output(ctx, {"collections": cols})
    else:
        log.header("Collections")
        for c in cols:
            log.info(f"  {c}")


@docstore.command("list", help="List documents in a collection")
@click.argument("collection")
@click.option("--limit", "-n", default=20, type=int)
@click.pass_context
def docstore_list(ctx, collection, limit):
    import requests
    timeout = ctx.obj.get("timeout", 10)
    r = requests.get(f"http://{ctx.obj['host']}:{ctx.obj['port']}/docstore/{collection}?limit={limit}",
                     timeout=timeout)
    if r.status_code != 200:
        log.error(f"Failed: {r.text}")
        sys.exit(1)
    data = r.json()
    docs = data.get("data", data).get("documents", [])
    if ctx.obj.get("json"):
        _output(ctx, {"documents": docs})
    else:
        log.header(f"{collection} ({len(docs)} docs)")
        for d in docs:
            doc_id = d.get("_id", d.get("id", "?"))
            log.info(f"  {doc_id}")


@docstore.command("get", help="Get a document")
@click.argument("collection")
@click.argument("doc_id")
@click.pass_context
def docstore_get(ctx, collection, doc_id):
    import requests
    timeout = ctx.obj.get("timeout", 10)
    r = requests.get(f"http://{ctx.obj['host']}:{ctx.obj['port']}/docstore/{collection}/{doc_id}",
                     timeout=timeout)
    if r.status_code != 200:
        log.error(f"Failed: {r.text}")
        sys.exit(1)
    data = r.json()
    doc = data.get("data", data)
    if ctx.obj.get("json"):
        _output(ctx, doc)
    else:
        for k, v in doc.items():
            log.info(f"  {k}: {v}")


@docstore.command("delete", help="Delete a document")
@click.argument("collection")
@click.argument("doc_id")
@click.option("--yes", "-y", is_flag=True)
@click.option("--dry-run", is_flag=True)
@click.pass_context
def docstore_delete(ctx, collection, doc_id, yes, dry_run):
    import requests
    if dry_run:
        log.info(f"Would delete {collection}/{doc_id}")
        return
    if not yes:
        confirm(f"Delete {collection}/{doc_id}?", abort=True)
    timeout = ctx.obj.get("timeout", 10)
    r = requests.delete(f"http://{ctx.obj['host']}:{ctx.obj['port']}/docstore/{collection}/{doc_id}",
                        timeout=timeout)
    if r.status_code == 200:
        log.success(f"Deleted {collection}/{doc_id}")
    else:
        log.error(f"Failed: {r.text}")
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════
# feeds  — rss, json
# ═══════════════════════════════════════════════════════════════════════


@cli.group(help="RSS and JSON feed generation from dev notes")
def feeds():
    pass


@feeds.command("rss", help="Generate RSS feed")
@click.option("--tag", default="", help="Filter by tag")
@click.option("--limit", "-n", default=20, type=int)
@click.option("--output", "-o", help="Save to file")
@click.pass_context
def feeds_rss(ctx, tag, limit, output):
    import requests
    timeout = ctx.obj.get("timeout", 10)
    params = {"limit": limit}
    if tag:
        params["tag"] = tag
    r = requests.get(f"http://{ctx.obj['host']}:{ctx.obj['port']}/feeds/rss.xml",
                     params=params, timeout=timeout, headers={"Accept": "application/xml"})
    if r.status_code != 200:
        log.error(f"RSS failed: {r.status_code}")
        sys.exit(1)
    content = r.text
    if output:
        with open(output, "w") as f:
            f.write(content)
        log.success(f"Saved to: {output}")
    else:
        print(content)


@feeds.command("json", help="Generate JSON feed")
@click.option("--tag", default="", help="Filter by tag")
@click.option("--limit", "-n", default=20, type=int)
@click.option("--output", "-o", help="Save to file")
@click.pass_context
def feeds_json(ctx, tag, limit, output):
    import requests
    timeout = ctx.obj.get("timeout", 10)
    params = {"limit": limit}
    if tag:
        params["tag"] = tag
    r = requests.get(f"http://{ctx.obj['host']}:{ctx.obj['port']}/feeds/feed.json",
                     params=params, timeout=timeout)
    if r.status_code != 200:
        log.error(f"JSON feed failed: {r.status_code}")
        sys.exit(1)
    content = r.text
    if output:
        with open(output, "w") as f:
            f.write(content)
        log.success(f"Saved to: {output}")
    else:
        print(content)


# ═══════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════


def main():
    # Parse command path for post-execution suggestions
    _argv = sys.argv[1:]
    _cmd_parts = []
    for _a in _argv:
        if _a.startswith("-"):
            break
        _cmd_parts.append(_a)
    _cmd_path = " ".join(_cmd_parts[:2])

    try:
        cli(obj={})
    except SystemExit:
        pass

    # Show post-command suggestions (TTY only)
    if _cmd_path and sys.stdout.isatty():
        _tip = _SUGGESTIONS.get(_cmd_path)
        if _tip:
            _p()
            _p(f"  {_c('💡', _DIM)} {_c('Tip:', _BOLD)} {_c(f'sloughgpt {_tip}', _CYAN)}")
            _p()


if __name__ == "__main__":
    main()
