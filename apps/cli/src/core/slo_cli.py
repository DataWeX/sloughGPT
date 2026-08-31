"""
slo_cli — Drop-in replacement for Click.

import slo_cli as click

Features:
- @group, @command, @option, @argument, @pass_context
- click.echo, click.confirm, click.Choice, click.Path
- Context with obj, ensure_object, invoke, invoked_subcommand
- Auto-correct with "Did you mean? [Y/n]" prompt
- Grouped, color-coded help output
- Post-command suggestions
- No external dependencies
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

def _p(text: str = "") -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


# ── Usage tracking ──────────────────────────────────────────────────────

_USAGE_PATH = _Path.home() / ".config" / "sloughgpt" / "usage_stats.json"

def _record_usage(cmd_name: str) -> None:
    try:
        data = {}
        if _USAGE_PATH.exists():
            data = json.loads(_USAGE_PATH.read_text())
        data[cmd_name] = data.get(cmd_name, 0) + 1
        _USAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _USAGE_PATH.write_text(json.dumps(data, indent=2))
    except Exception:
        pass


# ── Post-command suggestions ───────────────────────────────────────────

_SUGGESTIONS = {
    "model list": "model status",
    "model status": "model download",
    "model download": "model list",
    "dataset list": "dataset stats",
    "train dataset": "train start",
    "train start": "train monitor",
    "train monitor": "train eval",
    "system health": "system doctor",
    "system doctor": "system status",
    "checkpoint list": "checkpoint load",
    "memory stats": "memory search",
    "adapter list": "adapter info",
    "experiment list": "experiment info",
    "error recent": "error grouped",
    "feedback export": "feedback prepare",
    "token-tree train": "token-tree encode",
    "knowledge search": "knowledge gaps",
}


# ── Types ───────────────────────────────────────────────────────────────

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
        raise BadParameter(f"Invalid value '{value}' for {param_name}. Choose from: {valid}")


class Path:
    """Path type (like click.Path)."""

    def __init__(self, exists: bool = False, file_okay: bool = True,
                 dir_okay: bool = True, writable: bool = False,
                 readable: bool = True, resolve_path: bool = False):
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
            raise BadParameter(f"'{value}' does not exist")
        if p.exists():
            if not self.file_okay and p.is_file():
                raise BadParameter(f"'{value}' is a file, not a directory")
            if not self.dir_okay and p.is_dir():
                raise BadParameter(f"'{value}' is a directory, not a file")
        return value


# ── Exceptions ──────────────────────────────────────────────────────────

class UsageError(Exception):
    """Raised when a command is used incorrectly."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class BadParameter(UsageError):
    """Raised for invalid parameter values."""
    pass


class Abort(Exception):
    """Raised when user declines confirmation."""
    pass


# ── Parameter definitions ───────────────────────────────────────────────

class Option:
    """An option like --name VALUE or --flag."""

    def __init__(self, names: List[str], help: str = "", default: Any = None,
                 type: type = str, is_flag: bool = False, required: bool = False,
                 multiple: bool = False, metavar: str = "", show_default: bool = False,
                 choice: Optional[Choice] = None, flag_value: Any = None,
                 envvar: Optional[str] = None):
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
        self.flag_value = flag_value
        self.envvar = envvar
        self.dest = names[0].lstrip("-").replace("-", "_").replace("-", "_")

    @property
    def primary(self) -> str:
        return self.names[0]

    @property
    def secondary_names(self) -> List[str]:
        return self.names[1:]


class Argument:
    """A positional argument."""

    def __init__(self, name: str, required: bool = True, default: Any = None,
                 nargs: int = 1, type: Any = None):
        self.name = name
        self.required = required
        self.default = default
        self.nargs = nargs
        self.type = type


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
        self.callback = func


# ── Group ───────────────────────────────────────────────────────────────

class Group:
    """A command group with subcommands."""

    def __init__(self, name: str = "", help: str = "",
                 invoke_without_command: bool = False):
        self.name = name
        self.help = help
        self.commands: Dict[str, Command] = {}
        self.groups: Dict[str, "Group"] = {}
        self.parent: Optional["Group"] = None
        self.invoke_without_command = invoke_without_command
        self.callback: Optional[Callable] = None
        self._options: List[Option] = []
        self._arguments: List[Argument] = []

    def __call__(self, args=None, **kwargs):
        """Make Group callable — runs the CLI."""
        if args is None:
            args = sys.argv[1:]
        elif isinstance(args, str):
            args = args.split()
        run(self, list(args))

    def command(self, name: str = "", help: str = "", hidden: bool = False):
        """Decorator to register a command on this group."""
        def decorator(func):
            cmd_name = name or func.__name__
            # Use explicit help, or fall back to docstring
            cmd_help = help or (func.__doc__ or "").strip().split("\n")[0]
            cmd = Command(cmd_name, func, cmd_help, hidden=hidden)
            cmd.options = getattr(func, "_options", [])
            cmd.arguments = getattr(func, "_arguments", [])
            self.commands[cmd_name] = cmd
            return func
        return decorator

    def group(self, name: str = "", help: str = ""):
        """Decorator to register a subgroup on this group."""
        def decorator(func):
            grp_name = name or func.__name__
            grp = Group(grp_name, help)
            grp.callback = func
            grp.options = getattr(func, "_options", [])
            grp.arguments = getattr(func, "_arguments", [])
            self.groups[grp_name] = grp
            func._group = grp
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


# ── Context ─────────────────────────────────────────────────────────────

class Context:
    """Shared context passed to commands."""

    def __init__(self, obj: Optional[dict] = None, parent: Optional["Context"] = None):
        self.obj = obj or {}
        self.invoked_subcommand: Optional[str] = None
        self.parent = parent
        self._group: Optional[Group] = None
        self._command_path: List[str] = []

    def ensure_object(self, factory: Callable = dict):
        if not self.obj:
            self.obj = factory()

    def invoke(self, cmd, *args, **kwargs):
        """Invoke another command."""
        if isinstance(cmd, Command):
            cmd.func(*args, **kwargs)
        elif isinstance(cmd, Group):
            # Find a subcommand in the group
            pass


# ── Decorator helpers ───────────────────────────────────────────────────

def option(*names, help="", default=None, type=str, is_flag=False,
           required=False, multiple=False, metavar="", show_default=False,
           choice=None, flag_value=None, envvar=None, **kwargs):
    """Decorator to add an option to a command."""
    opt = Option(list(names), help=help, default=default, type=type,
                 is_flag=is_flag, required=required, multiple=multiple,
                 metavar=metavar, show_default=show_default, choice=choice,
                 flag_value=flag_value, envvar=envvar)
    def decorator(func):
        if not hasattr(func, "_options"):
            func._options = []
        func._options.append(opt)
        return func
    return decorator


def argument(name, required=True, default=None, nargs=1, type=None):
    """Decorator to add a positional argument to a command."""
    arg = Argument(name, required=required, default=default, nargs=nargs, type=type)
    def decorator(func):
        if not hasattr(func, "_arguments"):
            func._arguments = []
        func._arguments.append(arg)
        return func
    return decorator


def pass_context(func):
    """Decorator to mark that the command needs the context."""
    return func


def command(name: str = "", help: str = "", hidden: bool = False, **kwargs):
    """Decorator to create a standalone command (like @click.command)."""
    def decorator(func):
        cmd_name = name or func.__name__
        cmd_help = help or (func.__doc__ or "").strip().split("\n")[0]
        cmd = Command(cmd_name, func, cmd_help, hidden=hidden)
        cmd.options = getattr(func, "_options", [])
        cmd.arguments = getattr(func, "_arguments", [])
        return cmd
    return decorator


def version_option(package_name="", prog_name="", **kwargs):
    """Decorator to add --version option."""
    def decorator(func):
        if not hasattr(func, "_options"):
            func._options = []
        func._options.append(Option(
            ["--version"], help="Show version", is_flag=True,
            flag_value="version"
        ))
        return func
    return decorator


def confirmation_option(**kwargs):
    """Decorator to add --yes/-y option."""
    def decorator(func):
        if not hasattr(func, "_options"):
            func._options = []
        func._options.append(Option(
            ["--yes", "-y"], help="Skip confirmations", is_flag=True
        ))
        return func
    return decorator


# ── echo / confirm ──────────────────────────────────────────────────────

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
            raise Abort()
        return False
    if answer not in ("y", "yes"):
        if abort:
            raise Abort()
        return False
    return True


# ── Parser ──────────────────────────────────────────────────────────────

def _parse_args(args: List[str], options: List[Option], arguments: List[Argument]
               ) -> Tuple[dict, List[str]]:
    """Parse raw args into (kwargs, extra_args)."""
    kwargs = {}
    positional = []
    i = 0

    # Set defaults
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
            matched = False
            for opt in options:
                if arg in opt.names:
                    if opt.is_flag:
                        if arg.startswith("--no-"):
                            kwargs[opt.dest] = False
                        else:
                            kwargs[opt.dest] = True
                        matched = True
                        break
                    else:
                        i += 1
                        if i >= len(args):
                            raise BadParameter(f"Option {arg} requires a value")
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
                    raise BadParameter(f"Unknown option: {arg}")
        else:
            positional.append(arg)

        i += 1

    # Map positional args to argument definitions
    pos_idx = 0
    for arg_def in arguments:
        if arg_def.nargs == -1:
            kwargs[arg_def.name] = positional[pos_idx:]
            pos_idx = len(positional)
        elif pos_idx < len(positional):
            value = positional[pos_idx]
            if arg_def.type and isinstance(arg_def.type, Choice):
                value = arg_def.type.convert(value, arg_def.name)
            kwargs[arg_def.name] = value
            pos_idx += 1
        elif arg_def.required:
            raise BadParameter(f"Missing required argument: {arg_def.name}")
        else:
            kwargs[arg_def.name] = arg_def.default

    return kwargs, positional[pos_idx:]


# ── Help formatting ─────────────────────────────────────────────────────

def _format_help(group: Group, ctx: Context) -> str:
    """Generate grouped, color-coded help text."""
    try:
        from core.version import format_version_display
        version = format_version_display()
    except Exception:
        version = "dev"

    _p()
    _p(f"  {_c('SloughGPT', _BOLD + _CYAN)} {_c(f'({version})', _DIM)}")
    _p(f"  {_c('Train, chat, serve, and manage AI models', _DIM)}")
    _p()

    # Global options
    _p(f"  {_c('Global Options:', _BOLD)}")
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

    # Commands
    _p(f"  {_c('Commands:', _BOLD)}")

    printed = set()
    categories = {
        "Getting Started": ["start", "chat", "shell", "tui"],
        "Server": ["dev", "serve", "hf-serve"],
        "Models": ["model", "personality"],
        "Training": ["train", "checkpoint", "adapter", "feedback"],
        "Data": ["dataset", "knowledge", "experiment", "collect"],
        "Intelligence": ["tokenizer", "vector", "meta-weights", "learn", "memory", "token-tree"],
        "Media": ["images", "multimodal", "companion"],
        "System": ["system", "error", "completion", "simulate", "security", "docstore", "feeds", "logs", "monitor"],
        "Docker": ["docker", "build", "vm", "world"],
        "Advanced": ["agent", "session", "generate"],
    }

    for cat_name, cat_cmds in categories.items():
        items = []
        for cn in cat_cmds:
            if cn in group.commands:
                items.append(("cmd", group.commands[cn]))
            elif cn in group.groups:
                items.append(("grp", group.groups[cn]))
        if items:
            _p(f"\n    {_c(cat_name, _BOLD + _YELLOW)}")
            for kind, item in items:
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

    _p()
    _p(f"  {_c('Tips:', _BOLD)}")
    _p(f"    {_c('• Use fuzzy matching — \'sloughgpt md\' finds \'model\'', _DIM)}")
    _p(f"    {_c('• Add --yes/-y to skip confirmations', _DIM)}")
    _p()


def _format_group_help(group: Group, ctx: Context) -> None:
    """Show help for a subgroup."""
    _p()
    _p(f"  {_c(f'sloughgpt {group.name}', _BOLD + _CYAN)} — {group.help or ''}")
    _p()
    _p(f"  {_c('Commands:', _BOLD)}")
    for name in sorted(group.commands.keys()):
        cmd = group.commands[name]
        if cmd.hidden:
            continue
        padded = name.ljust(16)
        help_text = (cmd.help or "")[:50]
        _p(f"    {_c(padded, _CYAN)} {_c(help_text, _DIM)}")
    _p()
    _p(f"  {_c('Tip: Use \'--help\' for details on each command', _DIM)}")
    _p()


# ── Error display ───────────────────────────────────────────────────────

def _show_error(group: Group, cmd_name: str) -> None:
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


# ── Execution ───────────────────────────────────────────────────────────

def _run_command(cmd: Command, ctx: Context, args: List[str]) -> None:
    """Parse args and invoke a command."""
    try:
        kwargs, extra = _parse_args(args, cmd.options, cmd.arguments)
    except BadParameter as e:
        _p(f"  {_c('Error:', _RED)} {e}")
        sys.exit(1)

    # Check for --help
    if kwargs.get("help"):
        _p(f"\n  {_c(f'sloughgpt {ctx._command_path[-1] if ctx._command_path else cmd.name}', _BOLD + _CYAN)} — {cmd.help or ''}")
        _p()
        for opt in cmd.options:
            names = ", ".join(opt.names)
            help_text = opt.help or ""
            default_str = ""
            if opt.show_default and opt.default is not None:
                default_str = f" (default: {opt.default})"
            _p(f"    {_c(names, _CYAN)}   {_c(help_text + default_str, _DIM)}")
        for arg in cmd.arguments:
            _p(f"    {_c(arg.name, _CYAN)}   {_c('Required' if arg.required else 'Optional', _DIM)}")
        _p()
        return

    # Check for --version
    if kwargs.get("version"):
        try:
            from core.version import format_version_display
            echo(format_version_display())
        except Exception:
            echo("sloughgpt v0.3.0")
        return

    # Add context to kwargs if function accepts it
    import inspect
    sig = inspect.signature(cmd.func)
    if "ctx" in sig.parameters:
        kwargs["ctx"] = ctx

    try:
        cmd.func(**kwargs)
    except SystemExit:
        raise
    except Abort:
        _p(f"  {_c('Aborted.', _DIM)}")
        sys.exit(1)
    except KeyboardInterrupt:
        _p(f"\n  {_c('Interrupted', _DIM)}")
        sys.exit(130)
    except Exception as e:
        if not ctx.obj.get("quiet"):
            _p(f"  {_c('Error:', _RED)} {e}")
        sys.exit(1)


def _run_group(group: Group, ctx: Context, args: List[str]) -> None:
    """Run a group's subcommand."""
    # Parse group-level options
    group_kwargs, remaining = _parse_args(args, group.options, group.arguments)

    if not remaining:
        if group.invoke_without_command and group.callback:
            group_kwargs["ctx"] = ctx
            try:
                group.callback(**group_kwargs)
            except Exception:
                pass
            return
        _format_group_help(group, ctx)
        return

    cmd_name = remaining[0]
    cmd_args = remaining[1:]

    _resolve_and_run(group, ctx, cmd_name, cmd_args)


def _resolve_and_run(group: Group, ctx: Context, cmd_name: str, cmd_args: List[str]) -> None:
    """Resolve a command name and run it."""
    # Try exact match in groups
    if cmd_name in group.groups:
        sub = group.groups[cmd_name]
        _record_usage(cmd_name)
        ctx._command_path.append(cmd_name)
        _run_group(sub, ctx, cmd_args)
        return

    # Try exact match in commands
    if cmd_name in group.commands:
        cmd = group.commands[cmd_name]
        _record_usage(cmd_name)
        ctx._command_path.append(cmd_name)
        _run_command(cmd, ctx, cmd_args)
        return

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
                _resolve_and_run(group, ctx, best, cmd_args)
                return
            _show_error(group, cmd_name)
            sys.exit(1)
        else:
            _resolve_and_run(group, ctx, best, cmd_args)
            return

    _show_error(group, cmd_name)
    sys.exit(1)


# ── Top-level entry ─────────────────────────────────────────────────────

def run(group: Group, args: Optional[List[str]] = None) -> None:
    """Run the CLI. Main entry point."""
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
        if group.callback:
            try:
                group.callback(ctx=ctx, **ctx.obj)
            except Exception:
                pass
        if group.invoke_without_command:
            return
        _format_help(group, ctx)
        return

    cmd_name = remaining[0]
    cmd_args = remaining[1:]

    # Run the command
    _resolve_and_run(group, ctx, cmd_name, cmd_args)


def _parse_global_options(args: List[str]) -> Tuple[dict, List[str]]:
    """Extract global options from args."""
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


class _GroupDecorator:
    """Acts as both a decorator and a Group. Click compat."""

    def __init__(self, name: str, help: str, invoke_without_command: bool):
        self._group = Group(name, help, invoke_without_command=invoke_without_command)

    def __call__(self, func_or_none=None):
        if func_or_none is not None:
            self._group.callback = func_or_none
            return self._group
        return self

    def __getattr__(self, name):
        return getattr(self._group, name)

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._group, name, value)


# ── group() decorator ───────────────────────────────────────────────────

def group(name: str = "", help: str = "", invoke_without_command: bool = False,
          cls: type = None, **kwargs):
    """Decorator to create a command group (like @click.group)."""
    def decorator(func):
        grp = Group(name or func.__name__, help, invoke_without_command=invoke_without_command)
        grp.callback = func
        return grp
    return decorator
