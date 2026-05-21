#!/bin/bash
#
# SloughGPT Quick Setup
# Creates conda env + venv, installs PyTorch
#
set -e

CONDA_ENV="${CONDA_ENV:-sloughgpt}"
VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

echo "======================================"
echo "  SloughGPT Quick Setup"
echo "======================================"
echo ""
echo "Conda env: $CONDA_ENV"
echo "Venv dir:  $VENV_DIR"
echo "Python:    $PYTHON_VERSION"
echo ""

# Check for conda
if ! command -v conda &> /dev/null; then
    echo "Installing Miniconda..."
    brew install --cask miniconda
    eval "$(/opt/homebrew/bin/conda shell zsh hook)"
fi

# Init conda
eval "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true

# Create conda env
echo "Creating conda environment: $CONDA_ENV"
if conda env list | grep -q "^$CONDA_ENV "; then
    echo "Conda env already exists, skipping"
else
    conda create -n "$CONDA_ENV" python="$PYTHON_VERSION" -y
fi

# Install PyTorch in conda
echo ""
echo "Installing PyTorch in conda..."
conda activate "$CONDA_ENV"
if [[ "$(uname)" == "Darwin" ]]; then
    conda install pytorch torchvision torchaudio -c pytorch -y
else
    conda install pytorch torchvision torchaudio -c pytorch -y
fi

# Create venv
echo ""
echo "Creating venv: $VENV_DIR"
if [ -d "$VENV_DIR" ]; then
    echo "Venv already exists, skipping"
else
    python3 -m venv "$VENV_DIR"
fi

# Upgrade pip in venv
echo ""
echo "Upgrading pip..."
"$VENV_DIR/bin/python3" -m pip install --upgrade pip

# Install deps in venv (except torch)
echo ""
echo "Installing dependencies..."
"$VENV_DIR/bin/python3" -m pip install \
    numpy scipy pandas scikit-learn \
    fastapi uvicorn starlette httpx pydantic pydantic-settings \
    click rich prompt-toolkit \
    safetensors huggingface-hub accelerate \
    pillow requests tqdm pyyaml dataclasses-json \
    transformers \
    pytest pytest-asyncio ruff black mypy

# Install this package
echo ""
echo "Installing SloughGPT..."
"$VENV_DIR/bin/python3" -m pip install -e .

# Verify
echo ""
echo "======================================"
echo "  Verification"
echo "======================================"
echo ""

echo "PyTorch (conda):"
conda run -n "$CONDA_ENV" python3 -c "import torch; print(f'  torch: {torch.__version__}')"

echo ""
echo "SloughGPT (venv):"
"$VENV_DIR/bin/python3" -c "from domains.models import SloughGPTModel; print('  SloughGPTModel: OK')"

echo ""
echo "======================================"
echo "  Setup Complete!"
echo "======================================"
echo ""
echo "To activate conda:"
echo "  conda activate $CONDA_ENV"
echo ""
echo "To activate venv:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To run directly:"
echo "  conda run -n $CONDA_ENV python3 cli.py train --epochs 3"
echo "  $VENV_DIR/bin/python3 cli.py train --epochs 3"
echo ""
