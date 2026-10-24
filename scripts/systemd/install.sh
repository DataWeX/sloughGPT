#!/bin/bash
# Install the canonical slough-api.service unit into the user systemd dir.
#
# The unit lives in-repo (scripts/systemd/slough-api.service) so it is
# reviewable, versioned, and gated by tests/server/test_service_unit_env.py.
# ~/.config/systemd/user/ is a deploy artifact, not a place to hand-edit.
#
# Usage: scripts/systemd/install.sh [--dry-run] [--no-restart]
#
#   --dry-run     print what would change, touch nothing
#   --no-restart  install + daemon-reload, but leave the running service alone
#                 (use when other sessions are mid-request against :8000)
#
# Exit codes: 0 ok, 1 usage error, 2 live copy differs and --dry-run,
#             3 service failed to come back up after restart.

set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/slough-api.service"
DST="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/slough-api.service"
UNIT="slough-api.service"

DRY_RUN=0
NO_RESTART=0
for arg in "$@"; do
    case "$arg" in
        --dry-run)   DRY_RUN=1 ;;
        --no-restart) NO_RESTART=1 ;;
        -h|--help)   sed -n '2,16p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "install.sh: unknown flag '$arg' (try --help)" >&2; exit 1 ;;
    esac
done

[ -f "$SRC" ] || { echo "install.sh: no canonical unit at $SRC" >&2; exit 1; }

# The live copy is user data — never clobber it without a backup.
backup=""
if [ -f "$DST" ] && ! cmp -s "$SRC" "$DST"; then
    backup="$DST.bak.$(date +%Y%m%d-%H%M%S)"
fi

changed=0
[ "$backup" != "" ] && changed=1
[ ! -f "$DST" ] && changed=1

if [ "$DRY_RUN" = "1" ]; then
    if [ "$changed" = "1" ]; then
        echo "would install $SRC -> $DST"
        [ "$backup" != "" ] && echo "would back up existing to $backup"
        [ "$NO_RESTART" = "0" ] && echo "would restart $UNIT"
        exit 2
    fi
    echo "live copy already matches the canonical unit"
    exit 0
fi

mkdir -p "$(dirname "$DST")"
[ "$backup" != "" ] && { cp -p "$DST" "$backup"; echo "backed up -> $backup"; }
cp "$SRC" "$DST"
echo "installed $DST"

systemctl --user daemon-reload
systemctl --user enable "$UNIT" >/dev/null 2>&1 || true

if [ "$NO_RESTART" = "1" ]; then
    echo "--no-restart: daemon-reload done; run without the flag to apply"
    exit 0
fi

systemctl --user restart "$UNIT"

# systemd reports "active" as soon as the process is up; the API needs time to
# bind :8000, replay the mogdb journal and load the model. Measured startup on
# this box: ~15s idle, ~130s under load (2026-10-11) — so poll generously and
# let STARTUP_TIMEOUT override rather than having install.sh kill a healthy
# boot half-way and roll back over nothing.
STARTUP_TIMEOUT="${SLO_STARTUP_TIMEOUT:-300}"
ok=0
for _ in $(seq 1 $((STARTUP_TIMEOUT / 2))); do
    if [ "$(systemctl --user is-active "$UNIT" 2>/dev/null)" != "active" ]; then
        break
    fi
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8000/health 2>/dev/null || true)
    if [ "$code" = "200" ]; then ok=1; break; fi
    sleep 2
done

if [ "$ok" = "1" ]; then
    echo "OK: $UNIT active and /health 200"
    exit 0
fi

echo "FAILED: $UNIT did not serve /health after restart" >&2
journalctl --user -u "$UNIT" -n 20 --no-pager >&2 || true
[ "$backup" != "" ] && { echo "restoring $backup" >&2; cp -p "$backup" "$DST"; systemctl --user daemon-reload; systemctl --user restart "$UNIT" || true; }
exit 3
