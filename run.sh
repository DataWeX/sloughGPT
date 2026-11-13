#!/bin/bash
# Run any command with the project Python env active.
# Conda env 'sloughgpt' first, then the legacy repo .venv — same order as
# scripts/python, so activated-Python and pathed-Python never disagree.
# Example: ./run.sh python3 -m pytest tests/ -q
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"

# Activate the project conda env unless we are already in it. An active
# *other* env (e.g. `base`) must not short-circuit this check.
if [ "${CONDA_DEFAULT_ENV:-}" != "sloughgpt" ] && command -v conda >/dev/null 2>&1; then
  _conda_root="$(cd "$(dirname "$(command -v conda)")/.." && pwd)"
  if [ -f "$_conda_root/etc/profile.d/conda.sh" ] && [ -x "$_conda_root/envs/sloughgpt/bin/python" ]; then
    # shellcheck source=/dev/null
    . "$_conda_root/etc/profile.d/conda.sh"
    conda activate sloughgpt
  fi
fi

if [ -z "${CONDA_PREFIX:-}" ] && [ -f "$ROOT/.venv/bin/activate" ]; then
  # shellcheck source=/dev/null
  source "$ROOT/.venv/bin/activate"
fi

exec "$@"
