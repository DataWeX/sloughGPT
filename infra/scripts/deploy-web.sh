#!/usr/bin/env bash
set -euo pipefail

# Post-deploy script — runs on the server after rsync.
# Verifies the new build, reloads nginx, and runs a health check.
#
# Usage (via SSH stdin from the CI workflow):
#   ssh host 'bash -s' < deploy-web.sh
#
# Usage (on the server directly):
#   ./deploy-web.sh
#
# Environment variables:
#   WEB_ROOT       — web root directory (default: /var/www/html)
#   NGINX_SERVICE  — systemd service name (default: nginx)
#   HEALTH_PORT    — health-check port (default: 3000)

WEB_ROOT="${WEB_ROOT:-/var/www/html}"
NGINX_SERVICE="${NGINX_SERVICE:-nginx}"
HEALTH_PORT="${HEALTH_PORT:-3000}"

echo "=== Deploy verification ==="
echo "Web root: $WEB_ROOT"

if [ ! -f "$WEB_ROOT/index.html" ]; then
  echo "ERROR: index.html not found in $WEB_ROOT" >&2
  exit 1
fi

file_count=$(find "$WEB_ROOT" -type f | wc -l)
echo "Files deployed: $file_count"

echo "=== Reloading $NGINX_SERVICE ==="
if systemctl is-active --quiet "$NGINX_SERVICE" 2>/dev/null; then
  sudo systemctl reload "$NGINX_SERVICE"
  echo "$NGINX_SERVICE reloaded"
else
  sudo systemctl start "$NGINX_SERVICE"
  echo "$NGINX_SERVICE started (was not running)"
fi

echo "=== Health check ==="
sleep 2
if curl -sf "http://localhost:${HEALTH_PORT}/health" >/dev/null 2>&1; then
  echo "Health check: OK"
else
  echo "WARNING: health check failed" >&2
fi

echo "=== Deploy complete ==="
