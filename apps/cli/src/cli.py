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

# ── CLI framework (replaces Click) ───────────────────────────────────
from core.framework import (
    # types
    Option, Argument, Context, Command, Group,
    Choice, CliPath, IntRange, UsageError, BadParameter,
    # decorators
    group, command, option, argument, pass_context,
    version_option, confirmation_option, password_option,
    # output
    echo, confirm, p as _p, c as _c,
    # constants
    BOLD as _BOLD, DIM as _DIM, CYAN as _CYAN,
    GREEN as _GREEN, YELLOW as _YELLOW, RED as _RED,
    # dispatch
    run, record_usage as _record_usage,
    parse_args as _parse_args, run_command as _run_command,
    resolve_and_run as _resolve_and_run, run_group as _run_group,
    format_help as _format_help, format_group_help as _format_group_help,
    format_command_help as _format_command_help,
    # click namespace
    click,
)

# ── CLI helpers (Docker, banner, output) ─────────────────────────────
from core.helpers import (
    show_welcome_banner as _show_welcome_banner,
    show_server_status as _show_server_status,
    ns as _ns, output as _output, confirm as _confirm, verbose as _verbose,
)

_TTY = sys.stdout.isatty()  # re-export for backward compat


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
# model — list, inspect, download, export, and benchmark models
# ═══════════════════════════════════════════════════════════════════════

from groups.model import register as _register_model
_register_model(cli)

# ═══════════════════════════════════════════════════════════════════════
# dataset — list, import, export, and validate datasets
# ═══════════════════════════════════════════════════════════════════════

from groups.dataset import register as _register_dataset
_register_dataset(cli)

# ═══════════════════════════════════════════════════════════════════════
# train — training, evaluation, and monitoring
# ═══════════════════════════════════════════════════════════════════════

from groups.train import register as _register_train
_register_train(cli)

# ═══════════════════════════════════════════════════════════════════════
# token-tree — train, encode, decode, and query a tree tokenizer
# ═══════════════════════════════════════════════════════════════════════

from groups.token_tree import register as _register_token_tree
_register_token_tree(cli)

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
# knowledge — semantic knowledge operations
# ═══════════════════════════════════════════════════════════════════════

from groups.knowledge import register as _register_knowledge
_register_knowledge(cli)

# ═══════════════════════════════════════════════════════════════════════
# experiment — ML experiment tracking
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
# error — error monitoring
# ═══════════════════════════════════════════════════════════════════════

from groups.error import register as _register_error
_register_error(cli)

# ═══════════════════════════════════════════════════════════════════════
# memory — auto-memory layer management
# ═══════════════════════════════════════════════════════════════════════

from groups.memory import register as _register_memory
_register_memory(cli)

# ═══════════════════════════════════════════════════════════════════════
# personality — soul personality files
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
# agent — manage and execute AI agents
# ═══════════════════════════════════════════════════════════════════════

from groups.agent import register as _register_agent
_register_agent(cli)

# ═══════════════════════════════════════════════════════════════════════
# session — chat session management
# ═══════════════════════════════════════════════════════════════════════

from groups.session import register as _register_session
_register_session(cli)

# ═══════════════════════════════════════════════════════════════════════
# tokenizer — text tokenization
# ═══════════════════════════════════════════════════════════════════════

from groups.tokenizer import register as _register_tokenizer
_register_tokenizer(cli)

# ═══════════════════════════════════════════════════════════════════════
# vector — vector store for semantic search
# ═══════════════════════════════════════════════════════════════════════

from groups.vector import register as _register_vector
_register_vector(cli)

# ═══════════════════════════════════════════════════════════════════════
# system — system information and health
# ═══════════════════════════════════════════════════════════════════════

from groups.system import register as _register_system
_register_system(cli)

# ═══════════════════════════════════════════════════════════════════════
# docker — container management
# ═══════════════════════════════════════════════════════════════════════

from groups.docker import register as _register_docker
_register_docker(cli)

# ═══════════════════════════════════════════════════════════════════════
# simulate — boot kernel, load model, run inference, dump metrics
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


# ═══════════════════════════════════════════════════════════════════════
# world  — Render and simulate the programmable world
# ═══════════════════════════════════════════════════════════════════════

from groups.world import register as _register_world
_register_world(cli)

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
# db  — MogDB embedded database: migrate, sync, status
# ═══════════════════════════════════════════════════════════════════════


def _db_default_dir() -> str:
    return str(_chat_repository_root() / "data" / "mogdb")


def _import_mogdb():
    """Import mogdb, adding its src dir to sys.path for this session."""
    _src = Path(__file__).resolve().parents[3] / "packages" / "mogdb" / "src"
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))
    from mogdb import MogDB
    from mogdb.sync import sync_from_files, preview_from_files
    return MogDB, sync_from_files, preview_from_files


@cli.group(help="MogDB embedded database — migrate, sync, and status")
def db():
    pass


@db.command("migrate", help="Import a JSON/JSONL/CSV file into a MogDB collection")
@click.option("--file", required=True, help="Path to the source file (.json, .jsonl, or .csv)")
@click.option("--collection", help="Target collection name (default: from filename)")
@click.option("--db", default=None, help="MogDB data directory (default: repo data/mogdb)")
@click.option("--key", required=True, help="Identity/key field used for dedupe")
@click.option("--format", type=click.Choice(["json", "jsonl", "csv"]), help="Source format (default: auto-detect)")
@click.option("--delete-missing", is_flag=True, help="Delete collection docs missing from the file")
@click.option("--sync-dir", help="Write human-readable JSON sync files to this directory")
@click.option("--dry-run", is_flag=True, help="Report what would change without writing")
def db_migrate(file, collection, db, key, format, delete_missing, sync_dir, dry_run):
    from pathlib import Path as _Path
    _MogDB, _sync, _preview = _import_mogdb()
    path = _Path(file)
    if not path.exists():
        log.error(f"source file not found: {file}")
        sys.exit(2)
    if collection is None:
        collection = path.stem.replace("-", "_").replace(".", "_").lower()
    if db is None:
        db = _db_default_dir()
    _database = _MogDB(db, compact_on_close=not dry_run, sync_dir=sync_dir)
    try:
        _col = _database.collection(collection)
        result = (_preview if dry_run else _sync)(
            _col, str(path), key_field=key,
            delete_missing=delete_missing, file_format=format,
        )
    finally:
        _database.close()
    if dry_run:
        log.info(f"dry run for {collection}:")
    log.success(
        f"{collection}: +{result.inserted} ~{result.updated} -{result.deleted} ={result.unchanged}"
    )


@db.command("sync", help="Force JSON sync for all collections")
@click.option("--db", default=None, help="MogDB data directory (default: repo data/mogdb)")
@click.option("--sync-dir", required=True, help="Directory holding the JSON sync files")
def db_sync(db, sync_dir):
    _MogDB, _, _ = _import_mogdb()
    if db is None:
        db = _db_default_dir()
    _database = _MogDB(db, compact_on_close=True, sync_dir=sync_dir)
    try:
        names = _database.list_collections()
        # Loading a collection with sync_dir wraps it in SyncableCollection,
        # which immediately rewrites its JSON sync file.
        for name in names:
            _database.collection(name).sync()
    finally:
        _database.close()
    log.success(f"synced {len(names)} collections to {sync_dir}")


@db.command("status", help="Show MogDB collection stats")
@click.option("--db", default=None, help="MogDB data directory (default: repo data/mogdb)")
@click.option("--json", "as_json", is_flag=True, help="Print JSON output")
def db_status(db, as_json):
    _MogDB, _, _ = _import_mogdb()
    if db is None:
        db = _db_default_dir()
    _database = _MogDB(db, compact_on_close=False)
    try:
        names = _database.list_collections()
        rows = []
        for name in names:
            rows.append({"collection": name, "count": _database.collection(name).count()})
    finally:
        _database.close()
    if as_json:
        echo(json.dumps(rows, indent=2, default=str))
    else:
        for row in rows:
            echo(f"{row['collection']}: {row['count']} docs")


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
