#!/usr/bin/env bash
# E2E: real file → sidecar → gateway → downcraft (zstd) + Range resume.
#
# Verifies:
#   1. Full download through gateway matches source sha256
#   2. Wire savings (identity vs Accept-Encoding: zstd on the wire)
#   3. Stale-state reget re-downloads when dest is missing
#   4. Partial .sgpart resume uses Range → 206 → lossless
#
# Prereq: live stack (scripts/host_gateway.sh) OR set GW=http://127.0.0.1:8080
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-$REPO/.venv/bin/python}"
GW="${GW:-http://127.0.0.1:8080}"
SRC_REL="${SRC_REL:-checkpoints/step_25.pt}"
SRC="$REPO/$SRC_REL"
OUT="${OUT:-/tmp/e2e_gw_dc.pt}"
WORKDIR="$(dirname "$OUT")"

fail() { echo "E2E_FAIL: $*" >&2; exit 1; }
ok() { echo "E2E_OK: $*"; }

# ── 0. stack health ────────────────────────────────────────────────────────
curl -sf "$GW/health" >/dev/null || fail "gateway not healthy at $GW/health"
[ -f "$SRC" ] || fail "source missing: $SRC"

SRC_SHA="$(sha256sum "$SRC" | awk '{print $1}')"
SRC_LEN="$(wc -c < "$SRC")"
echo "source: $SRC_REL  $SRC_LEN bytes  sha256=${SRC_SHA:0:16}…"

# ── 1. wire savings: identity vs zstd through the gateway ─────────────────
ID_BYTES=$(curl -s -o /tmp/e2e_id.bin -w '%{size_download}' \
  -H 'Accept-Encoding: identity' "$GW/$SRC_REL")
ZSTD_BYTES=$(curl -s -o /tmp/e2e_zs.bin -w '%{size_download}' \
  -H 'Accept-Encoding: zstd' "$GW/$SRC_REL")
GZIP_BYTES=$(curl -s -o /tmp/e2e_gz.bin -w '%{size_download}' \
  -H 'Accept-Encoding: gzip' "$GW/$SRC_REL")

# identity payload must match source
cmp -s /tmp/e2e_id.bin "$SRC" || fail "identity body != source"
# decode compressed and compare
"$PY" - <<'PY'
import io, pathlib, gzip, zstandard as zstd
src = pathlib.Path("/tmp/e2e_id.bin").read_bytes()
zs = pathlib.Path("/tmp/e2e_zs.bin").read_bytes()
with zstd.ZstdDecompressor().stream_reader(io.BytesIO(zs)) as r:
    assert r.read() == src, "zstd lossless mismatch"
assert gzip.decompress(pathlib.Path("/tmp/e2e_gz.bin").read_bytes()) == src, "gzip lossless mismatch"
print("decoded zstd+gzip match identity: OK")
PY

ZSTD_SAVINGS=$(awk -v a="$ID_BYTES" -v b="$ZSTD_BYTES" 'BEGIN{printf "%.1f", (1-b/a)*100}')
GZIP_SAVINGS=$(awk -v a="$ID_BYTES" -v b="$GZIP_BYTES" 'BEGIN{printf "%.1f", (1-b/a)*100}')
echo "wire: identity=$ID_BYTES zstd=$ZSTD_BYTES (${ZSTD_SAVINGS}% saved) gzip=$GZIP_BYTES (${GZIP_SAVINGS}% saved)"

# ── 2. full downcraft download (zstd path) ────────────────────────────────
rm -f "$OUT" "$OUT.sgpart"
# clear state for this URL so we don't short-circuit
"$PY" - <<PY
import json, pathlib
p = pathlib.Path.home() / ".downcraft/state.json"
if not p.exists():
    raise SystemExit(0)
st = json.loads(p.read_text())
models = st.get("models") or {}
key = "$GW/$SRC_REL"
if key in models:
    models.pop(key, None)
    p.write_text(json.dumps(st, indent=2))
    print("cleared state for", key)
PY

"$PY" -m downcraft url "$GW/$SRC_REL" "$OUT" >/tmp/e2e_full.log 2>&1 \
  || { cat /tmp/e2e_full.log; fail "full download"; }
OUT_SHA="$(sha256sum "$OUT" | awk '{print $1}')"
[ "$OUT_SHA" = "$SRC_SHA" ] || fail "full download sha256 mismatch"
ok "full download sha256 match ($SRC_LEN bytes)"

# ── 3. reget: dest deleted, state still complete ──────────────────────────
rm -f "$OUT"
"$PY" -m downcraft url "$GW/$SRC_REL" "$OUT" >/tmp/e2e_reget.log 2>&1 \
  || { cat /tmp/e2e_reget.log; fail "reget download"; }
grep -q "re-downloading" /tmp/e2e_reget.log || fail "reget did not notice missing dest"
OUT_SHA="$(sha256sum "$OUT" | awk '{print $1}')"
[ "$OUT_SHA" = "$SRC_SHA" ] || fail "reget sha256 mismatch"
ok "reget after deleted dest"

# ── 4. Range resume from 1 MiB partial ────────────────────────────────────
rm -f "$OUT" "$OUT.sgpart"
PART=$((1024 * 1024))
dd if="$SRC" of="$OUT.sgpart" bs=1M count=1 status=none
"$PY" -m downcraft url "$GW/$SRC_REL" "$OUT" >/tmp/e2e_resume.log 2>&1 \
  || { cat /tmp/e2e_resume.log; fail "resume download"; }
grep -q "Resuming" /tmp/e2e_resume.log || fail "resume did not send Range"
grep -q "doesn't support Range" /tmp/e2e_resume.log && fail "gateway returned non-206 for Range"
OUT_SHA="$(sha256sum "$OUT" | awk '{print $1}')"
[ "$OUT_SHA" = "$SRC_SHA" ] || fail "resume sha256 mismatch"
ok "Range resume from $PART bytes (206)"

echo ""
echo "=== E2E summary ==="
echo "  file:       $SRC_REL ($SRC_LEN B)"
echo "  identity:   $ID_BYTES B wire"
echo "  zstd:       $ZSTD_BYTES B wire  (${ZSTD_SAVINGS}% saved)"
echo "  gzip:       $GZIP_BYTES B wire  (${GZIP_SAVINGS}% saved)"
echo "  full/reget/resume: all sha256 = ${SRC_SHA:0:16}…"
echo "E2E_DONE"
