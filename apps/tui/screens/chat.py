"""Chat screen for SloughGPT TUI — thin shell wrapper.

Every input goes through ``ShellREPL.execute()``. The TUI is a
rendering layer — the shell is the engine. No duplicate API calls,
no duplicate state management.
"""

from __future__ import annotations

from rich.text import Text

from apps.tui.components import CONSOLE, Color, divider
from apps.tui.session import TuiSession
from apps.tui.screen import Screen
from apps.tui.bindings import Binding


class ChatScreen(Screen):
    name = "chat"
    bindings = [Binding(["/"], "clear", "clear")]

    def __init__(self):
        super().__init__()
        self.input_buffer: str = ""
        self._repl = None

    def _get_repl(self):
        """Lazy-init ShellREPL — imports core-py on first use."""
        if self._repl is None:
            import sys
            from pathlib import Path
            repo = Path(__file__).resolve().parents[3]
            core_py = str(repo / "packages" / "core-py")
            if core_py not in sys.path:
                sys.path.insert(0, core_py)
            from domains.shell import get_dait_runtime, ShellREPL
            from domains.shell.commands import ShellCommands
            os = get_dait_runtime()
            cmds = ShellCommands()
            self._repl = ShellREPL(os, cmds)
        return self._repl

    def render(self, session: TuiSession) -> str:
        self.render_header("Shell", session.api_base_url)
        divider()
        CONSOLE.print()

        if self.input_buffer:
            CONSOLE.print(Text(f"  {self.input_buffer}", style=Color.HIGHLIGHT))
        else:
            CONSOLE.print(Text("  Type a command — shell handles everything", style=Color.MUTED))
        CONSOLE.print()
        CONSOLE.print(Text(self.binding_manager.format_footer(self.bindings), style=Color.MUTED))

        return self._handle_input(session)

    def _handle_input(self, session: TuiSession) -> str:
        import readchar

        while True:
            key = readchar.readkey()

            for b in self.binding_manager.global_bindings:
                if key in b.keys:
                    return b.action

            if key == readchar.key.ENTER:
                text = self.input_buffer.strip()
                if not text:
                    return "chat"

                self.input_buffer = ""

                if text.lower() in ("exit", "quit", "q"):
                    return "quit"

                repl = self._get_repl()
                output, exit_code = repl.execute(text)

                if output:
                    CONSOLE.print()
                    for line in output.rstrip("\n").split("\n"):
                        CONSOLE.print(Text(f"  {line}", style=Color.WHITE))

                return "chat"

            elif key == "/":
                self.input_buffer = ""
                return "chat"

            elif key == readchar.key.BACKSPACE:
                self.input_buffer = self.input_buffer[:-1]
                return "chat"

            elif len(key) == 1 and key.isprintable():
                self.input_buffer += key
                return "chat"

            return "chat"
