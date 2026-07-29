import sys
from .core import cli_main as notes_cli_main
from .kanban import cli_main as kanban_cli_main

if len(sys.argv) > 1 and sys.argv[1] == "kanban":
    sys.exit(kanban_cli_main(sys.argv[2:]))
sys.exit(notes_cli_main(sys.argv[1:] if len(sys.argv) > 1 else None))
