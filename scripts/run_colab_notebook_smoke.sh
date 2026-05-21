#!/usr/bin/env bash
# Execute sloughgpt_colab.ipynb end-to-end with smoke-friendly defaults.
# Uses notebook env hooks: SLOUGH_NOTEBOOK_FORCE_CPU, SLOUGH_NOTEBOOK_TRAIN_CAP (see §3 / §7).
# Output: sloughgpt_colab.executed.ipynb in repo root (gitignored).
# §13 / trainer step_*.pt charset for cli.py eval: docs/policies/CONTRIBUTING.md (Checkpoint vocabulary).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

case "${1:-}" in
  -h|--help)
    cat <<'EOF'
Usage: run_colab_notebook_smoke.sh [--help]

Runs nbconvert --execute on sloughgpt_colab.ipynb (`jupyter nbconvert` or
`python3 -m nbconvert`); writes sloughgpt_colab.executed.ipynb in the repo root (gitignored).

Smoke-oriented environment (override as needed):
  SLOUGH_NOTEBOOK_FORCE_CPU   default: 1 (set 0 for GPU/MPS)
  SLOUGH_NOTEBOOK_TRAIN_CAP   default: 50 (unset for full §7 batches; e.g. 1 for quickest local verify)

Install: python3 -m pip install -e ".[notebook]"
Docs: README.md → "Google Colab"

Windows: use Git Bash or WSL so bash can run this script (.sh).
EOF
    exit 0
    ;;
esac

export SLOUGH_NOTEBOOK_FORCE_CPU="${SLOUGH_NOTEBOOK_FORCE_CPU:-1}"
export SLOUGH_NOTEBOOK_TRAIN_CAP="${SLOUGH_NOTEBOOK_TRAIN_CAP:-50}"

_run_nbconvert() {
  if command -v jupyter >/dev/null 2>&1; then
    jupyter nbconvert "$@"
  elif python3 -m nbconvert --help >/dev/null 2>&1; then
    python3 -m nbconvert "$@"
  else
    echo "Install nbconvert: python3 -m pip install -e \".[notebook]\"  (or jupyter nbconvert)" >&2
    return 1
  fi
}

_run_nbconvert \
  --to notebook \
  --execute \
  --ExecutePreprocessor.timeout=7200 \
  --output sloughgpt_colab.executed.ipynb \
  sloughgpt_colab.ipynb
