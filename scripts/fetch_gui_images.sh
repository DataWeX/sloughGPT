#!/usr/bin/env bash
# Fetch the GUI desktop saved-states from the copy.sh demo CDN into
# apps/web/public/gui/ (gitignored — vendored per session, never served from
# a runtime CDN).
#
# Why a script and not a browser fetch: c.copy.sh answers every cross-origin
# browser request with 403 (hotlink protection); only a plain server-side
# request gets through. The states are full machine snapshots (memory +
# devices + sparse disk), which is how the copy.sh demo keeps these desktops
# bootable even though their raw images (windows98/.img, reactos-v3/.img, …)
# 404 today. Each state MUST be restored with the memory size it was saved
# with (see apps/web/public/gui/README.md table).
#
# Verified alive (HTTP 200/206) 2026-10-19. Re-run any time; present files
# are skipped, so it is cheap and idempotent.
set -euo pipefail

BASE="https://c.copy.sh"
OUT="$(cd "$(dirname "$0")/.." && pwd)/apps/web/public/gui"
mkdir -p "$OUT"

fetch() {
  local f="$1"
  local dest="$OUT/$f"
  if [[ -s "$dest" ]]; then
    echo "skip  $f (present)"
    return
  fi
  echo "get   $f"
  curl -fL --max-time 900 --retry 2 --retry-delay 5 -o "$dest.part" "$BASE/$f"
  mv "$dest.part" "$dest"
}

# GUI desktop states (id -> state file, memory the snapshot was saved with):
#   windows98    Windows 98        128 MB   windows98_state-v2.bin.zst     12.8 MB
#   windows-me   Windows ME        256 MB   windows-me_state-v3.bin.zst    27.7 MB
#   windows2000  Windows 2000      512 MB   windows2k_state-v4.bin.zst     28.2 MB
#   reactos      ReactOS           512 MB   reactos_state-v3.bin.zst       17.5 MB
#   serenity     SerenityOS        512 MB   serenity_state-v4.bin.zst      16.2 MB
#   haiku        Haiku             512 MB   haiku_state-v5.bin.zst         36.5 MB
#   redox        Redox            1024 MB   redox_state-v2.bin.zst         15.8 MB
#   9front       9front            128 MB   9front_state-v3.bin.zst         5.2 MB
fetch windows98_state-v2.bin.zst
fetch windows-me_state-v3.bin.zst
fetch windows2k_state-v4.bin.zst
fetch reactos_state-v3.bin.zst
fetch serenity_state-v4.bin.zst
fetch haiku_state-v5.bin.zst
fetch redox_state-v2.bin.zst
fetch 9front_state-v3.bin.zst

echo
ls -la "$OUT"
