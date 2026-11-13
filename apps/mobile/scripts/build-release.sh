#!/usr/bin/env bash
# Build release APK
# Usage: ./scripts/build-release.sh

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Pre-build checks ==="

# TypeScript
echo "TypeScript..."
npx tsc --noEmit
echo "  ✓ Clean"

# Tests
echo "Tests..."
npx jest --no-coverage --silent
echo "  ✓ Passed"

# Clean
echo "Cleaning..."
cd android
./gradlew clean

# Build
echo "Building release APK..."
./gradlew assembleRelease

APK="app/build/outputs/apk/release/app-release.apk"
if [ -f "$APK" ]; then
  SIZE=$(du -h "$APK" | cut -f1)
  echo ""
  echo "=== Build complete ==="
  echo "APK: $APK ($SIZE)"
  echo "Install: adb install $APK"
else
  echo "Build failed — APK not found"
  exit 1
fi
