"""SloughGPT TUI — Interactive keyboard-driven terminal UI.

Uses polymorphic dispatch: each screen is a ``Screen`` subclass instance.
The event loop calls ``screen.render(session)`` and routes to the next screen.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from rich.text import Text
from apps.tui.components import CONSOLE, Color
from apps.tui.session import TuiSession, discover_repo_root

from apps.tui.screens.home import HomeScreen
from apps.tui.screens.chat import ChatScreen
from apps.tui.screens.generate import GenerateScreen
from apps.tui.screens.models import ModelsScreen
from apps.tui.screens.dataset import DatasetScreen
from apps.tui.screens.eval import EvalScreen
from apps.tui.screens.status import StatusScreen
from apps.tui.screens.train import TrainScreen
from apps.tui.screens.about import AboutScreen
from apps.tui.screens.palette import PaletteScreen


SCREENS = {
    "home": HomeScreen(),
    "chat": ChatScreen(),
    "generate": GenerateScreen(),
    "models": ModelsScreen(),
    "dataset": DatasetScreen(),
    "eval": EvalScreen(),
    "status": StatusScreen(),
    "train": TrainScreen(),
    "about": AboutScreen(),
    "palette": PaletteScreen(),
}


def _goodbye():
    CONSOLE.clear()
    CONSOLE.print()
    CONSOLE.print("  Goodbye!", style=Color.PRIMARY)
    CONSOLE.print()


def run_interactive(session: TuiSession):
    """Main event loop — routes between screens via SCREENS dict.

    Saves session state on clean exit so the next launch restores
    the last checkpoint, soul, job, etc.
    """
    screen_name = "home"

    while True:
        try:
            if screen_name == "__palette__":
                result = SCREENS["palette"].render(session)
                screen_name = result if result else screen_name

            screen = SCREENS.get(screen_name) if screen_name not in ("__palette__", "quit") else None

            if screen is not None:
                screen_name = screen.render(session)

            elif screen_name == "quit":
                session.save()
                _goodbye()
                return

            else:
                screen_name = "home"

        except KeyboardInterrupt:
            session.save()
            _goodbye()
            return
        except Exception as e:
            CONSOLE.print()
            CONSOLE.print(Text(f"  Error: {e}", style=f"bold {Color.ERROR}"))
            if sys.stdin.isatty():
                CONSOLE.print(Text("  Press any key to continue...", style=Color.MUTED))
                import readchar
                try:
                    readchar.readkey()
                except Exception:
                    pass
            screen_name = "home"


def main(argv: Optional[list] = None) -> None:
    import argparse

    args = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(prog="sloughgpt-tui")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--interactive", action="store_true")
    ns = parser.parse_args(args)

    root = ns.repo_root.resolve() if ns.repo_root else (discover_repo_root() or Path.cwd())

    session = TuiSession.load(repo_root=root)
    if ns.host != "127.0.0.1":
        session.api_host = ns.host
    if ns.port != 8000:
        session.api_port = ns.port

    try:
        run_interactive(session)
    except KeyboardInterrupt:
        _goodbye()


if __name__ == "__main__":
    main()
