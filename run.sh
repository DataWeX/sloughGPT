#!/bin/bash
# Run any command with the repo .venv activated (when present).
# Example: ./run.sh python3 -m pytest tests/ -q
#          ./run.sh python3 cli.py train --epochs 1
#          ./run.sh ./scripts/run_colab_notebook_smoke.sh   # after: pip install -e ".[notebook]"
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$ROOT/.venv/bin/activate" ]; then
  source "$ROOT/.venv/bin/activate"
fi
exec "$@"
