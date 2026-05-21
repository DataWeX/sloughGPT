"""SloughGPT TUI — launch interactive mode."""

import sys

from apps.tui.interactive import main

if __name__ == "__main__":
    main(sys.argv[1:])
