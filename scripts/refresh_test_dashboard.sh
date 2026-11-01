#!/usr/bin/env bash
# refresh_test_dashboard.sh — run tests, publish HTML to web public/
# View at http://localhost:3000/test_dashboard.html (or any static host)
set -euo pipefail
cd "$(dirname "$0")/.."

./scripts/python scripts/test_dashboard.py "$@"
cp test_dashboard.html apps/web/public/test_dashboard.html
echo "Published: apps/web/public/test_dashboard.html"
