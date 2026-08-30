"""
SmartGroup — Click Group with fuzzy matching and polished error output.

Provides better "did you mean?" suggestions, color-coded help, and
a welcome banner for the CLI.
"""

import sys
import click
from difflib import get_close_matches
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

def _p(text: str = "") -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


class SmartGroup(click.Group):
    """Click Group with fuzzy command matching and polished error output.

    When a user types a command that doesn't exist, this group:
    1. Checks for exact substring matches first
    2. Falls back to fuzzy matching (difflib)
    3. Shows a helpful error with suggestions
    4. Displays available commands grouped by category
    """

    def get_command(self, ctx: click.Context, cmd_name: str) -> Optional[click.Command]:
        """Get command with fuzzy matching fallback."""
        # Try exact match first
        cmd = super().get_command(ctx, cmd_name)
        if cmd is not None:
            return cmd

        # Try fuzzy matching
        matches = self._fuzzy_match(cmd_name)
        if matches:
            # Return the best match
            return super().get_command(ctx, matches[0])

        return None

    def _fuzzy_match(self, cmd_name: str) -> List[str]:
        """Find close matches for a command name."""
        # Get all available commands
        commands = list(self.commands.keys())

        # First: try substring matching
        substring_matches = [c for c in commands if cmd_name.lower() in c.lower()]
        if substring_matches:
            return substring_matches

        # Second: try fuzzy matching with difflib
        close = get_close_matches(cmd_name, commands, n=3, cutoff=0.6)
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
                "cmds": ["dataset", "knowledge", "experiment"],
                "desc": "Import and manage training data",
            },
            "System": {
                "cmds": ["system", "error", "completion", "simulate"],
                "desc": "Environment and diagnostics",
            },
            "Docker": {
                "cmds": ["docker"],
                "desc": "Containerized deployment",
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
