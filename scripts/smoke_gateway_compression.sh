#!/usr/bin/env bash
# Live smoke: sidecar → gateway → curl identity/gzip/zstd/no-accept-encoding.
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="$REPO/.venv/bin/python"
GW="$REPO/apps/gateway/target/debug/slough-gateway"

cleanup() {
  [ -n "${GW_PID:-}" ] && kill "$GW_PID" 2>/dev/null
  [ -n "${SIDECAR_PID:-}" ] && kill "$SIDECAR_PID" 2>/dev/null
  wait 2>/dev/null
}
trap cleanup EXIT

"$PY" -c '
from http.server import BaseHTTPRequestHandler, HTTPServer
PAYLOAD = (b"id=1 name=item status=ok padding " * 40000)
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(PAYLOAD)))
        self.end_headers()
        self.wfile.write(PAYLOAD)
    def log_message(self, *a): pass
HTTPServer(("127.0.0.1", 18000), H).serve_forever()
' &
SIDECAR_PID=$!
sleep 0.5

if ! curl -sf -o /dev/null http://127.0.0.1:18000/files.dat; then
  echo "sidecar FAILED to start"; exit 1
fi

if [ ! -x "$GW" ]; then
  (cd "$REPO/apps/gateway" && cargo build --quiet)
fi

MAN_CORE_URL=http://127.0.0.1:18000 MAN_GATEWAY_PORT=18080 "$GW" &
GW_PID=$!
for _ in $(seq 1 40); do
  curl -sf http://127.0.0.1:18080/health >/dev/null 2>&1 && break
  sleep 0.25
done

echo "=== identity ==="
curl -s -o /tmp/id.bin -D /tmp/id.hdr -H "Accept-Encoding: identity" http://127.0.0.1:18080/files.dat
grep -iE "HTTP/|content-encoding|content-length" /tmp/id.hdr | tr -d '\r'
echo "wire: $(wc -c < /tmp/id.bin)"

echo "=== gzip ==="
curl -s -o /tmp/gz.raw -D /tmp/gz.hdr -H "Accept-Encoding: gzip" http://127.0.0.1:18080/files.dat
grep -iE "HTTP/|content-encoding|content-length|vary:" /tmp/gz.hdr | tr -d '\r'
echo "wire: $(wc -c < /tmp/gz.raw)"
"$PY" -c "
import gzip, pathlib
dec = gzip.decompress(pathlib.Path('/tmp/gz.raw').read_bytes())
orig = pathlib.Path('/tmp/id.bin').read_bytes()
assert dec == orig, 'gzip not lossless'
print('lossless: True decoded:', len(dec))
"

echo "=== zstd ==="
curl -s -o /tmp/zs.raw -D /tmp/zs.hdr -H "Accept-Encoding: zstd" http://127.0.0.1:18080/files.dat
grep -iE "HTTP/|content-encoding|content-length|vary:" /tmp/zs.hdr | tr -d '\r'
echo "wire: $(wc -c < /tmp/zs.raw)"
"$PY" -c "
import io, pathlib, zstandard as zstd
raw = pathlib.Path('/tmp/zs.raw').read_bytes()
with zstd.ZstdDecompressor().stream_reader(io.BytesIO(raw)) as r:
    dec = r.read()
orig = pathlib.Path('/tmp/id.bin').read_bytes()
assert dec == orig, 'zstd not lossless'
print('lossless: True decoded:', len(dec))
"

echo "=== no Accept-Encoding → identity passthrough ==="
curl -s -o /tmp/none.bin -D /tmp/none.hdr http://127.0.0.1:18080/files.dat
grep -iE "content-encoding|content-length" /tmp/none.hdr | tr -d '\r' || true
if cmp -s /tmp/none.bin /tmp/id.bin; then
  echo "byte-identical passthrough: OK"
else
  echo "PASSTHROUGH MISMATCH"; exit 1
fi

echo SMOKE_DONE
