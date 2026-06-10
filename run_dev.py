#!/usr/bin/env python3
"""
SloughGPT Dev Server — starts API + web frontend.

Usage:
    python run_dev.py              # legacy runner (delegates to `sloughgpt serve --web`)
    python run_dev.py --model gpt2 # preload a model
    sloughgpt serve --web          # preferred way
"""

import sys
import subprocess
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    python = sys.executable

    args = ["-m", "apps.cli.src.cli", "serve", "--web"]
    extra = [a for a in sys.argv[1:] if a not in ("dev", "--web")]
    if extra:
        args.extend(extra)

    proc = subprocess.run([python, *args], cwd=root)
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
