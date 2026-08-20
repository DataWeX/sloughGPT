#!/bin/bash
# Fix CMake paths with spaces: "Default\ Project" -> "Default Project"
find app/build -name "*.cmake" -o -name "CMakeLists.txt" | while read f; do
  if grep -q 'Default\\ Project' "$f" 2>/dev/null; then
    sed -i 's|Default\\ Project|Default Project|g' "$f"
  fi
done
