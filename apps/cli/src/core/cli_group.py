"""
SmartGroup — Click Group with fuzzy matching and polished error output.

Provides better "did you mean?" suggestions, color-coded help, and
a welcome banner for the CLI.
"""

import click
from difflib import get_close_matches
from typing import List, Optional


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
        from rich.console import Console
        from rich.text import Text

        console = Console(highlight=False)

        # Get available commands
        commands = sorted(self.commands.keys())

        # Build error message
        error = Text()
        error.append("  ✗ ", style="red")
        error.append(f"Unknown command: ", style="bold")
        error.append(cmd_name, style="cyan")

        console.print()
        console.print(error)
        console.print()

        # Show suggestions
        if commands:
            console.print("  [bold]Available commands:[/]")
            for name in commands:
                console.print(f"    [cyan]{name}[/]")
            console.print()

            # Show tip
            console.print("  [dim]Tip: Use 'sloughgpt --help' to see all commands[/]")
            console.print()

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
        # Title
        from rich.console import Console
        console = Console(highlight=False)

        # Print version and title
        try:
            from core.version import format_version_display
            version = format_version_display()
        except Exception:
            version = "dev"

        console.print(f"\n  [bold cyan]SloughGPT[/] [dim]({version})[/]\n")

        # Description
        console.print("  [dim]Train, chat, serve, and manage AI models[/]\n")

        # Commands grouped by category with examples and tips
        self._format_grouped_commands(ctx, console)

        # Global options
        console.print("  [bold]Global Options:[/]")
        console.print("    [cyan]--host[/]       API hostname (default: localhost)")
        console.print("    [cyan]--port[/]       API port (default: 8000)")
        console.print("    [cyan]-c, --config[/] Config path (default: config.yaml)")
        console.print("    [cyan]--yes, -y[/]    Skip all confirmations")
        console.print("    [cyan]--help[/]       Show this help message")
        console.print()

    def _format_grouped_commands(self, ctx: click.Context, console) -> None:
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
                "cmds": ["dataset", "knowledge"],
                "desc": "Import and manage training data",
            },
            "System": {
                "cmds": ["system", "completion", "simulate"],
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
        console.print("  [bold]Commands:[/]")

        # Track which commands we've printed
        printed = set()

        for cat_name, cat_info in categories.items():
            cat_cmds = [(n, commands[n]) for n in cat_info["cmds"] if n in commands]
            if cat_cmds:
                console.print(f"\n    [bold yellow]{cat_name}[/] [dim]— {cat_info['desc']}[/]")
                for name, cmd in cat_cmds:
                    help_text = cmd.get_short_help_str(limit=50)
                    if help_text:
                        console.print(f"      [cyan]{name:<16}[/] [dim]{help_text}[/]")
                    else:
                        console.print(f"      [cyan]{name}[/]")
                    printed.add(name)

        # Print any remaining commands (hidden ones like hf-serve)
        remaining = [(n, commands[n]) for n in sorted(commands.keys()) if n not in printed]
        if remaining:
            console.print("\n    [bold yellow]Advanced[/]")
            for name, cmd in remaining:
                if getattr(cmd, "hidden", False):
                    continue
                help_text = cmd.get_short_help_str(limit=50)
                if help_text:
                    console.print(f"      [cyan]{name:<16}[/] [dim]{help_text}[/]")
                else:
                    console.print(f"      [cyan]{name}[/]")

        # Print examples
        console.print("\n  [bold]Examples:[/]")
        console.print("    [dim]sloughgpt chat[/]                     Start chatting")
        console.print("    [dim]sloughgpt model download gpt2[/]     Download a model")
        console.print("    [dim]sloughgpt model status[/]             Check model cache")
        console.print("    [dim]sloughgpt train dataset shakespeare[/] Train on dataset")
        console.print("    [dim]sloughgpt shell[/]                    Interactive shell")
        console.print()

        # Print tips
        console.print("  [bold]Tips:[/]")
        console.print("    [dim]• Use fuzzy matching — 'sloughgpt md' finds 'model'[/]")
        console.print("    [dim]• Run 'sloughgpt shell' then 'confirm on' to skip all download prompts[/]")
        console.print("    [dim]• Run 'sloughgpt shell' for 40+ built-in commands[/]")
        console.print("    [dim]• Add --yes/-y to skip confirmations for a single command[/]")
        console.print()

        console.print()

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
