#!/usr/bin/env bash
# Bump version across package.json, build.gradle, and commit
# Usage: ./scripts/bump-version.sh [major|minor|patch]

set -euo pipefail

BUMP="${1:-patch}"
cd "$(dirname "$0")/.."

# Read current version
CURRENT=$(node -p "require('./package.json').version")
IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT"

case "$BUMP" in
  major) MAJOR=$((MAJOR + 1)); MINOR=0; PATCH=0 ;;
  minor) MINOR=$((MINOR + 1)); PATCH=0 ;;
  patch) PATCH=$((PATCH + 1)) ;;
  *) echo "Usage: $0 [major|minor|patch]"; exit 1 ;;
esac

NEW="$MAJOR.$MINOR.$PATCH"
echo "Bumping $CURRENT → $NEW"

# Update package.json
sed -i "s/\"version\": \"$CURRENT\"/\"version\": \"$NEW\"/" package.json

# Update Android build.gradle
sed -i "s/versionCode [0-9]*/versionCode $((MAJOR * 10000 + MINOR * 100 + PATCH))/" android/app/build.gradle
sed -i "s/versionName \"$CURRENT\"/versionName \"$NEW\"/" android/app/build.gradle

echo "Updated: package.json, build.gradle"
echo "Version: $NEW (code: $((MAJOR * 10000 + MINOR * 100 + PATCH)))"
