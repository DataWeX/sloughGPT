"""
SloughGPT CLI Framework — Click-API-compatible, zero Click dependency.

Reimplements Click's decorator API (group, command, option, argument, etc.)
using pure Python. The rest of the CLI uses this as if it were Click.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from difflib import get_close_matches
from typing import Any, Callable, Dict, List, Optional, Tuple

_TTY = sys.stdout.isatty()

# ── ANSI helpers ──────────────────────────────────────────────────────

BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"

def c(text: str, code: str) -> str:
    """Wrap text in ANSI color codes (no-op if not TTY)."""
    return f"{code}{text}\033[0m" if _TTY else text

# ── Usage tracking ───────────────────────────────────────────────────

_USAGE_PATH = Path.home() / ".config" / "sloughgpt" / "usage_stats.json"

def record_usage(cmd_name: str) -> None:
    try:
        _USAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        if _USAGE_PATH.exists():
            data = json.loads(_USAGE_PATH.read_text())
        data[cmd_name] = data.get(cmd_name, 0) + 1
        _USAGE_PATH.write_text(json.dumps(data, indent=2))
    except Exception:
        pass

# ── Output helpers ───────────────────────────────────────────────────

def echo(message: str = "", nl: bool = True, err: bool = False):
    stream = sys.stderr if err else sys.stdout
    if message:
        stream.write(str(message))
    if nl:
        stream.write("\n")
    stream.flush()

def p(text: str = "") -> None:
    """Print a line to stdout."""
    sys.stdout.write(text + "\n")
    sys.stdout.flush()

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

# ── Type validators ──────────────────────────────────────────────────

class Choice:
    def __init__(self, choices: List[str], case_sensitive: bool = False):
        self.choices = choices
        self.case_sensitive = case_sensitive
    def convert(self, value: str, param_name: str) -> str:
        if self.case_sensitive:
            if value in self.choices:
                return value
        else:
            for ch in self.choices:
                if ch.lower() == value.lower():
                    return ch
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
        p_ = Path(value)
        if self.resolve_path:
            p_ = p_.resolve()
            value = str(p_)
        if self.exists and not p_.exists():
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

# ── Exceptions ───────────────────────────────────────────────────────

class UsageError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)

class BadParameter(UsageError):
    pass

# ── Core types ───────────────────────────────────────────────────────

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
        prefix = [ch for ch in all_names if ch.startswith(cmd_name.lower())]
        if prefix:
            return prefix
        substring = [ch for ch in all_names if cmd_name.lower() in ch.lower()]
        if substring:
            return substring
        cutoff = 0.4 if len(cmd_name) <= 3 else 0.6
        return get_close_matches(cmd_name, all_names, n=3, cutoff=cutoff)
    def __call__(self, *args, **kwargs):
        run(self)

# ── Decorators ───────────────────────────────────────────────────────

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

# ── Argument parsing ─────────────────────────────────────────────────

def parse_args(args: List[str], options: List[Option], arguments: List[Argument]
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

# ── Help formatting ──────────────────────────────────────────────────

def format_help(group: Group, ctx: Context,
                categories: Optional[Dict] = None,
                suggestions: Optional[Dict] = None,
                version_fn: Optional[Callable] = None) -> None:
    try:
        version = version_fn() if version_fn else "dev"
    except Exception:
        version = "dev"
    p()
    p(f"  {c('SloughGPT', BOLD + CYAN)} {c(f'({version})', DIM)}")
    p(f"  {c('Train, chat, serve, and manage AI models', DIM)}")
    p()
    p(f"  {c('Commands:', BOLD)}")
    printed = set()
    if categories:
        for cat_name, cat_info in categories.items():
            cat_items = []
            for cmd_name in cat_info["cmds"]:
                if cmd_name in group.commands:
                    cat_items.append(("cmd", group.commands[cmd_name]))
                elif cmd_name in group.groups:
                    cat_items.append(("grp", group.groups[cmd_name]))
            if cat_items:
                desc = cat_info["desc"]
                p(f"\n    {c(cat_name, BOLD + YELLOW)} {c(f'— {desc}', DIM)}")
                for kind, item in cat_items:
                    if kind == "cmd" and item.hidden:
                        continue
                    padded = item.name.ljust(16)
                    help_text = item.help or ""
                    p(f"      {c(padded, CYAN)} {c(help_text, DIM)}")
                    printed.add(item.name)
    remaining = []
    for name in sorted(group.commands.keys()):
        if name not in printed and not group.commands[name].hidden:
            remaining.append(("cmd", group.commands[name]))
    for name in sorted(group.groups.keys()):
        if name not in printed:
            remaining.append(("grp", group.groups[name]))
    if remaining:
        p(f"\n    {c('Other', BOLD + YELLOW)}")
        for kind, item in remaining:
            padded = item.name.ljust(16)
            help_text = item.help or ""
            p(f"      {c(padded, CYAN)} {c(help_text, DIM)}")
    p(f"\n  {c('Global Options:', BOLD)}")
    p(f"    {c('--host', CYAN)}         API hostname (default: localhost)")
    p(f"    {c('--port', CYAN)}         API port (default: 8000)")
    p(f"    {c('-c, --config', CYAN)}   Config path (default: config.yaml)")
    p(f"    {c('--json', CYAN)}         JSON output for commands")
    p(f"    {c('--no-color', CYAN)}     Disable ANSI color output")
    p(f"    {c('-q, --quiet', CYAN)}    Suppress non-essential output")
    p(f"    {c('--timeout', CYAN)}      HTTP timeout in seconds (default: 10)")
    p(f"    {c('--version', CYAN)}      Show version")
    p(f"    {c('--yes, -y', CYAN)}      Skip all confirmations")
    p(f"    {c('--help', CYAN)}         Show this help message")
    p()
    p(f"  {c('Examples:', BOLD)}")
    p(f"    sloughgpt chat                     Start chatting")
    p(f"    sloughgpt model download gpt2     Download a model")
    p(f"    sloughgpt model status             Check model cache")
    p(f"    sloughgpt train dataset shakespeare Train on dataset")
    p(f"    sloughgpt shell                    Interactive shell")
    p()
    p(f"  {c('Tips:', BOLD)}")
    p(f"    {c('•', GREEN)} Use fuzzy matching — {c("'sloughgpt md'", CYAN)} finds {c('model', CYAN)}")
    p(f"    {c('•', GREEN)} Run {c("'sloughgpt shell'", CYAN)} then {c("'confirm on'", CYAN)} to skip all download prompts")
    p(f"    {c('•', GREEN)} Run {c("'sloughgpt shell'", CYAN)} for 40+ built-in commands")
    p(f"    {c('•', GREEN)} Add {c('--yes/-y', CYAN)} to skip confirmations for a single command")
    p()

def format_group_help(group: Group, ctx: Context) -> None:
    p()
    p(f"  {c('Usage:', BOLD)} {group.name} [OPTIONS] COMMAND [ARGS]...")
    if group.help:
        p(f"\n  {group.help}")
    p()
    if group.commands:
        p(f"  {c('Commands:', BOLD)}")
        for name in sorted(group.commands.keys()):
            cmd = group.commands[name]
            if cmd.hidden:
                continue
            padded = name.ljust(16)
            help_text = cmd.help or ""
            p(f"    {c(padded, CYAN)} {c(help_text, DIM)}")
        p()
    p(f"  {c('--help', CYAN)}   Show this help message")
    p()

def format_command_help(cmd: Command, ctx: Context, cmd_path: str = "") -> None:
    p()
    name = cmd_path or cmd.name
    p(f"  {c('Usage:', BOLD)} sloughgpt {name} [OPTIONS]")
    if cmd.help:
        p()
        p(f"  {cmd.help}")
    if cmd.options:
        p()
        p(f"  {c('Options:', BOLD)}")
        for opt in cmd.options:
            names = ", ".join(opt.names)
            default_str = ""
            if opt.default is not None and opt.default != "" and not opt.is_flag:
                default_str = f" {c(f'(default: {opt.default})', DIM)}"
            required_str = f" {c('(required)', RED)}" if opt.required else ""
            p(f"    {names:<20} {opt.help}{default_str}{required_str}")
    p()

# ── Command dispatch ─────────────────────────────────────────────────

def run_command(cmd: Command, ctx: Context, args: List[str], cmd_path: str = "") -> None:
    if "--help" in args or "-h" in args:
        format_command_help(cmd, ctx, cmd_path)
        return
    try:
        kwargs, extra = parse_args(args, cmd.options, cmd.arguments)
    except UsageError as e:
        p(f"  {c('Error:', RED)} {e}")
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
        p(f"\n  {c('Interrupted', DIM)}")
        sys.exit(130)
    except UsageError as e:
        p(f"  {c('Error:', RED)} {e}")
        sys.exit(1)
    except Exception as e:
        if not ctx.obj.get("quiet"):
            p(f"  {c('Error:', RED)} {e}")
        sys.exit(1)

def _show_error(group: Group, cmd_name: str) -> None:
    p()
    p(f"  {c('err', RED)} {c('Unknown command: ', BOLD)}{c(cmd_name, CYAN)}")
    p()
    all_names = sorted(list(group.commands.keys()) + list(group.groups.keys()))
    if all_names:
        p(f"  {c('Available commands:', BOLD)}")
        for name in all_names:
            p(f"    {c(name, CYAN)}")
        p()
        p(f"  {c('Tip: Use \'sloughgpt --help\' to see all commands', DIM)}")
        p()

def resolve_and_run(group: Group, ctx: Context, cmd_name: str, cmd_args: List[str],
                    full_path: str = "") -> Optional[Tuple[str, List[str]]]:
    if cmd_name in group.groups:
        sub = group.groups[cmd_name]
        record_usage(cmd_name)
        return run_group(sub, ctx, cmd_args, full_path=f"{full_path} {cmd_name}" if full_path else cmd_name)
    if cmd_name in group.commands:
        cmd = group.commands[cmd_name]
        record_usage(cmd_name)
        full = f"{full_path} {cmd_name}" if full_path else cmd_name
        run_command(cmd, ctx, cmd_args, cmd_path=full)
        return (full, cmd_args)
    matches = group._fuzzy_match(cmd_name)
    if matches:
        best = matches[0]
        if _TTY and sys.stdin.isatty():
            p()
            p(f"  {c('?', YELLOW)} {c('Unknown command: ', DIM)}{c(cmd_name, RED)}")
            p(f"  {c('>', GREEN)} {c('Did you mean ', DIM)}{c(best, CYAN + BOLD)}{c('?', DIM)}")
            try:
                answer = input("    [Y/n] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = "n"
            if answer in ("", "y", "yes"):
                return resolve_and_run(group, ctx, best, cmd_args, full_path)
            _show_error(group, cmd_name)
            return None
        else:
            return resolve_and_run(group, ctx, best, cmd_args, full_path)
    _show_error(group, cmd_name)
    return None

def run_group(group: Group, ctx: Context, args: List[str],
              full_path: str = "") -> Optional[Tuple[str, List[str]]]:
    if group.invoke_without_command:
        cb = group.callback if hasattr(group, 'callback') else None
        if cb:
            import inspect
            sig = inspect.signature(cb)
            grp_opts = getattr(group, "_options", [])
            try:
                kwargs, positional = parse_args(args, grp_opts, [])
            except UsageError as e:
                p(f"  {c('Error:', RED)} {e}")
                sys.exit(1)
            if "ctx" in sig.parameters:
                kwargs["ctx"] = ctx
            cb(**kwargs)
            args = positional
    if not args:
        if not group.invoke_without_command:
            format_group_help(group, ctx)
        return None
    cmd_name = args[0]
    cmd_args = args[1:]
    path = f"{full_path} {cmd_name}" if full_path else cmd_name
    return resolve_and_run(group, ctx, cmd_name, cmd_args, full_path=full_path)

def run(group: Group, args: Optional[List[str]] = None,
        categories: Optional[Dict] = None,
        suggestions: Optional[Dict] = None,
        version_fn: Optional[Callable] = None):
    """Single entry-point for CLI dispatch."""
    if args is None:
        args = sys.argv[1:]

    ctx = Context()
    ctx.ensure_object()

    grp_opts = list(getattr(group, "_options", [])) + [
        Option(["--help", "-h"],  help="Show this message and exit", is_flag=True),
        Option(["--version"],     help="Show version and exit",      is_flag=True),
    ]
    try:
        kwargs, positional = parse_args(args, grp_opts, [])
    except UsageError as e:
        p(f"  {c('Error:', RED)} {e}")
        sys.exit(1)

    show_help = kwargs.pop("help", False)
    show_version = kwargs.pop("version", False)

    cmd_name = positional[0] if positional else None
    cmd_args = positional[1:] if len(positional) > 1 else []

    ctx.obj.update(kwargs)
    if cmd_name:
        ctx.invoked_subcommand = cmd_name

    if hasattr(group, "callback") and group.callback:
        import inspect
        sig = inspect.signature(group.callback)
        cb_kwargs = dict(kwargs)
        if "ctx" in sig.parameters:
            cb_kwargs["ctx"] = ctx
        group.callback(**cb_kwargs)

    if show_help:
        format_help(group, ctx, categories=categories, suggestions=suggestions, version_fn=version_fn)
        return
    if show_version:
        try:
            echo(version_fn() if version_fn else "sloughgpt v0.1.0")
        except Exception:
            echo("sloughgpt v0.1.0")
        return

    if cmd_name is None:
        format_help(group, ctx, categories=categories, suggestions=suggestions, version_fn=version_fn)
        return

    result = resolve_and_run(group, ctx, cmd_name, cmd_args)
    if _TTY and result and suggestions:
        suggestion = suggestions.get(result[0])
        if suggestion:
            p(f"\n  {c(suggestion, DIM)}")

# ── Click-compatible namespace ───────────────────────────────────────

click = SimpleNamespace(
    group=group, command=command, option=option, argument=argument,
    pass_context=pass_context, version_option=version_option,
    confirmation_option=confirmation_option, echo=echo, confirm=confirm,
    Choice=Choice, Path=CliPath, UsageError=UsageError, BadParameter=BadParameter,
    run=run,
)
