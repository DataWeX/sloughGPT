"""
SmartGroup — Click Group with fuzzy matching and polished error output.

Provides better "did you mean?" suggestions, color-coded help, and
a welcome banner for the CLI.
"""

import sys
import json
import click
from difflib import get_close_matches
from pathlib import Path
from typing import List, Optional


# ── ANSI helpers ────────────────────────────────────────────────────────

_TTY = sys.stdout.isatty()

def _c(text: str, code: str) -> str:
    return f"{code}{text}\033[0m" if _TTY else text

_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"

_USAGE_FILE = Path.home() / ".config" / "sloughgpt" / "usage_stats.json"

def _p(text: str = "") -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


# ── Post-command suggestions ────────────────────────────────────────────

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


class SmartGroup(click.Group):
    """Click Group with fuzzy command matching and polished error output.

    When a user types a command that doesn't exist, this group:
    1. Checks for exact substring matches first
    2. Falls back to fuzzy matching (difflib)
    3. Shows a helpful error with suggestions
    4. Displays available commands grouped by category
    """

    def get_command(self, ctx: click.Context, cmd_name: str) -> Optional[click.Command]:
        """Get command with fuzzy matching and auto-correct prompt."""
        # Try exact match first
        cmd = super().get_command(ctx, cmd_name)
        if cmd is not None:
            return cmd

        # Try fuzzy matching
        matches = self._fuzzy_match(cmd_name)
        if matches:
            best = matches[0]
            # Prompt for auto-correction if TTY
            if _TTY and sys.stdin.isatty():
                _p()
                _p(f"  {_c('?', _YELLOW)} {_c('Unknown command: ', _DIM)}{_c(cmd_name, _RED)}")
                _p(f"  {_c('→', _GREEN)} {_c('Did you mean ', _DIM)}{_c(best, _CYAN + _BOLD)}{_c('?', _DIM)}")
                try:
                    answer = input("    [Y/n] ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    answer = "n"
                if answer in ("", "y", "yes"):
                    return super().get_command(ctx, best)
                # User declined — show full error
                self._show_command_error(ctx, cmd_name)
                ctx.exit(1)
                return None
            else:
                # Non-interactive: auto-resolve silently
                return super().get_command(ctx, best)

        return None

    def resolve_command(self, ctx: click.Context, args: List[str]) -> tuple:
        """Override to add auto-correct prompt for mistyped commands."""
        # Let Click try to resolve normally
        try:
            return super().resolve_command(ctx, args)
        except click.UsageError:
            # Command not found — try fuzzy matching
            if args:
                cmd_name = args[0]
                matches = self._fuzzy_match(cmd_name)
                if matches:
                    best = matches[0]
                    # Prompt for auto-correction if TTY
                    if _TTY and sys.stdin.isatty():
                        _p()
                        _p(f"  {_c('?', _YELLOW)} {_c('Unknown command: ', _DIM)}{_c(cmd_name, _RED)}")
                        _p(f"  {_c('→', _GREEN)} {_c('Did you mean ', _DIM)}{_c(best, _CYAN + _BOLD)}{_c('?', _DIM)}")
                        try:
                            answer = input("    [Y/n] ").strip().lower()
                        except (EOFError, KeyboardInterrupt):
                            answer = "n"
                        if answer in ("", "y", "yes"):
                            # Replace the bad command with the good one
                            return super().resolve_command(ctx, [best] + args[1:])
                    else:
                        # Non-interactive: auto-resolve silently
                        return super().resolve_command(ctx, [best] + args[1:])
            # Re-raise if no match or user declined
            raise

    def _fuzzy_match(self, cmd_name: str) -> List[str]:
        """Find close matches for a command name."""
        commands = list(self.commands.keys())

        # Exact prefix match (highest priority — "md" matches "model")
        prefix_matches = [c for c in commands if c.startswith(cmd_name.lower())]
        if prefix_matches:
            return prefix_matches

        # Substring match
        substring_matches = [c for c in commands if cmd_name.lower() in c.lower()]
        if substring_matches:
            return substring_matches

        # Fuzzy match with lowered cutoff for short inputs
        cutoff = 0.4 if len(cmd_name) <= 3 else 0.6
        close = get_close_matches(cmd_name, commands, n=3, cutoff=cutoff)
        return close

    def _show_command_error(self, ctx: click.Context, cmd_name: str) -> None:
        """Show a helpful error message when command is not found."""
        # Get available commands
        commands = sorted(self.commands.keys())

        _p()
        _p(f"  {_c('✗', _RED)} {_c('Unknown command: ', _BOLD)}{_c(cmd_name, _CYAN)}")
        _p()

        # Show suggestions
        if commands:
            _p(f"  {_c('Available commands:', _BOLD)}")
            for name in commands:
                _p(f"    {_c(name, _CYAN)}")
            _p()

            # Show tip
            _p(f"  {_c('Tip: Use \'sloughgpt --help\' to see all commands', _DIM)}")
            _p()

    def format_usage(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """Custom usage format with version info."""
        # Get the program name
        prog = ctx.find_root().info_name or ctx.info_name

        # Get the command path
        if ctx.parent:
            path = ctx.parent.command_path
            if ctx.info_name:
                path += f" {ctx.info_name}"
        else:
            path = ctx.command_path

        # Format usage line
        formatter.write_usage(path, "[OPTIONS] COMMAND [ARGS]...")

    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """Custom help format with color-coded groups and examples."""
        # Print version and title
        try:
            from core.version import format_version_display
            version = format_version_display()
        except Exception:
            version = "dev"

        _p(f"\n  {_c('SloughGPT', _BOLD + _CYAN)} {_c(f'({version})', _DIM)}\n")

        # Description
        _p(f"  {_c('Train, chat, serve, and manage AI models', _DIM)}\n")

        # Commands grouped by category with examples and tips
        self._format_grouped_commands(ctx)

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

    def _format_grouped_commands(self, ctx: click.Context) -> None:
        """Format commands grouped by category with examples and tips."""
        # Define command categories with descriptions
        categories = {
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

        # Get all commands
        commands = {}
        for name in self.list_commands(ctx):
            cmd = self.get_command(ctx, name)
            if cmd is not None:
                commands[name] = cmd

        # Print grouped commands
        _p(f"  {_c('Commands:', _BOLD)}")

        # Track which commands we've printed
        printed = set()

        for cat_name, cat_info in categories.items():
            cat_cmds = [(n, commands[n]) for n in cat_info["cmds"] if n in commands]
            if cat_cmds:
                desc = cat_info["desc"]
                _p(f"\n    {_c(cat_name, _BOLD + _YELLOW)} {_c(f'— {desc}', _DIM)}")
                for name, cmd in cat_cmds:
                    help_text = cmd.get_short_help_str(limit=50)
                    padded = name.ljust(16)
                    if help_text:
                        _p(f"      {_c(padded, _CYAN)} {_c(help_text, _DIM)}")
                    else:
                        _p(f"      {_c(name, _CYAN)}")
                    printed.add(name)

        # Print any remaining commands (hidden ones like hf-serve)
        remaining = [(n, commands[n]) for n in sorted(commands.keys()) if n not in printed]
        if remaining:
            _p(f"\n    {_c('Advanced', _BOLD + _YELLOW)}")
            for name, cmd in remaining:
                if getattr(cmd, "hidden", False):
                    continue
                help_text = cmd.get_short_help_str(limit=50)
                padded = name.ljust(16)
                if help_text:
                    _p(f"      {_c(padded, _CYAN)} {_c(help_text, _DIM)}")
                else:
                    _p(f"      {_c(name, _CYAN)}")

        # Print examples
        _p(f"\n  {_c('Examples:', _BOLD)}")
        _p(f"    {_c('sloughgpt chat', _DIM)}                     Start chatting")
        _p(f"    {_c('sloughgpt model download gpt2', _DIM)}     Download a model")
        _p(f"    {_c('sloughgpt model status', _DIM)}             Check model cache")
        _p(f"    {_c('sloughgpt train dataset shakespeare', _DIM)} Train on dataset")
        _p(f"    {_c('sloughgpt shell', _DIM)}                    Interactive shell")
        _p()

        # Print tips
        _p(f"  {_c('Tips:', _BOLD)}")
        _p(f"    {_c('• Use fuzzy matching — \'sloughgpt md\' finds \'model\'', _DIM)}")
        _p(f"    {_c('• Run \'sloughgpt shell\' then \'confirm on\' to skip all download prompts', _DIM)}")
        _p(f"    {_c('• Run \'sloughgpt shell\' for 40+ built-in commands', _DIM)}")
        _p(f"    {_c('• Add --yes/-y to skip confirmations for a single command', _DIM)}")
        _p()
        _p()

    def resolve_command(self, ctx: click.Context, args: List[str]) -> tuple:
        """Resolve command with fuzzy matching and helpful error on failure."""
        # Try to resolve normally
        try:
            return super().resolve_command(ctx, args)
        except click.UsageError as e:
            # If it's a "no such command" error, try to provide better suggestions
            if "No such command" in str(e) and args:
                cmd_name = args[0]
                matches = self._fuzzy_match(cmd_name)

                if matches:
                    # We found a match, try again
                    return super().resolve_command(ctx, args)
                else:
                    # No match found, show helpful error
                    self._show_command_error(ctx, cmd_name)
                    raise SystemExit(1)
            raise
        except SystemExit:
            raise
        except Exception as e:
            # Catch any other exceptions and show a clean error
            if hasattr(e, '__traceback__'):
                # This is likely a Click error
                raise
            # For other errors, show a clean message
            self._show_command_error(ctx, args[0] if args else "")
            raise SystemExit(1)

    def invoke(self, ctx: click.Context) -> None:
        """Track usage and show post-command suggestions."""
        cmd_path = ctx.command_path.replace("cli.py ", "")
        self._track_usage(cmd_path)
        super().invoke(ctx)

    def _track_usage(self, cmd_path: str) -> None:
        """Track command usage in ~/.config/sloughgpt/usage_stats.json."""
        try:
            _USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
            stats = {}
            if _USAGE_FILE.exists():
                stats = json.loads(_USAGE_FILE.read_text())
            stats[cmd_path] = stats.get(cmd_path, 0) + 1
            _USAGE_FILE.write_text(json.dumps(stats, indent=2))
        except Exception:
            pass
