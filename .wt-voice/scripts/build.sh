#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$ROOT/build"
DIST_DIR="$ROOT/dist"
VENV_PYTHON="${PYTHON:-python3}"

echo "==> Building sloughgpt binaries"
echo "    Python: $($VENV_PYTHON --version)"
echo "    PyInstaller: $($VENV_PYTHON -m PyInstaller --version)"
echo "    Target: $DIST_DIR"
echo ""

mkdir -p "$DIST_DIR"

build_one() {
    local name="$1"
    local spec="$BUILD_DIR/$name.spec"
    echo "==> Building $name..."
    $VENV_PYTHON -m PyInstaller \
        --workpath "$BUILD_DIR/.build-$name" \
        --specpath "$BUILD_DIR" \
        --distpath "$DIST_DIR" \
        --clean \
        "$spec"
    echo "    Done: $DIST_DIR/$name"
    echo ""
}

build_one "sloughgpt"

echo "==> All builds complete!"
echo "    Binaries:"
ls -lh "$DIST_DIR/sloughgpt" 2>/dev/null || \
ls -lh "$DIST_DIR/" 2>/dev/null
