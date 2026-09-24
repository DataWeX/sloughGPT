#!/usr/bin/env bash
# Live smoke: sidecar → gateway → curl identity/gzip/zstd/no-accept-encoding/Range.
# Alternate ports so it can run beside a live host_gateway stack.
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="$REPO/.venv/bin/python"
GW="$REPO/apps/gateway/target/debug/slough-gateway"
SIDECAR_PORT="${SIDECAR_PORT:-18010}"
GW_PORT="${GW_PORT:-18090}"

cleanup() {
  [ -n "${GW_PID:-}" ] && kill "$GW_PID" 2>/dev/null
  [ -n "${SIDECAR_PID:-}" ] && kill "$SIDECAR_PID" 2>/dev/null
  wait 2>/dev/null
}
trap cleanup EXIT

"$PY" -c "
from http.server import BaseHTTPRequestHandler, HTTPServer
PAYLOAD = (b'id=1 name=item status=ok padding ' * 40000)
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        rng = self.headers.get('Range')
        if rng and rng.startswith('bytes='):
            spec = rng[6:].split(',')[0].strip()
            a, _, b = spec.partition('-')
            start = int(a) if a else 0
            end = min(int(b), len(PAYLOAD)-1) if b else len(PAYLOAD)-1
            if start > end or start >= len(PAYLOAD):
                self.send_response(416)
                self.send_header('Content-Range', f'bytes */{len(PAYLOAD)}')
                self.send_header('Content-Length', '0')
                self.end_headers()
                return
            body = PAYLOAD[start:end+1]
            self.send_response(206)
            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('Content-Range', f'bytes {start}-{end}/{len(PAYLOAD)}')
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header('Content-Type', 'application/octet-stream')
        self.send_header('Content-Length', str(len(PAYLOAD)))
        self.send_header('Accept-Ranges', 'bytes')
        self.end_headers()
        self.wfile.write(PAYLOAD)
    def log_message(self, *a): pass
HTTPServer(('127.0.0.1', $SIDECAR_PORT), H).serve_forever()
" &
SIDECAR_PID=$!
sleep 0.5

if ! curl -sf -o /dev/null "http://127.0.0.1:$SIDECAR_PORT/files.dat"; then
  echo "sidecar FAILED to start"; exit 1
fi

if [ ! -x "$GW" ]; then
  (cd "$REPO/apps/gateway" && cargo build --quiet)
fi

MAN_CORE_URL="http://127.0.0.1:$SIDECAR_PORT" MAN_GATEWAY_PORT="$GW_PORT" "$GW" &
GW_PID=$!
for _ in $(seq 1 40); do
  curl -sf "http://127.0.0.1:$GW_PORT/health" >/dev/null 2>&1 && break
  sleep 0.25
done

echo "=== identity ==="
curl -s -o /tmp/id.bin -D /tmp/id.hdr -H "Accept-Encoding: identity" "http://127.0.0.1:$GW_PORT/files.dat"
grep -iE "HTTP/|content-encoding|content-length" /tmp/id.hdr | tr -d '\r'
echo "wire: $(wc -c < /tmp/id.bin)"

echo "=== gzip ==="
curl -s -o /tmp/gz.raw -D /tmp/gz.hdr -H "Accept-Encoding: gzip" "http://127.0.0.1:$GW_PORT/files.dat"
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
curl -s -o /tmp/zs.raw -D /tmp/zs.hdr -H "Accept-Encoding: zstd" "http://127.0.0.1:$GW_PORT/files.dat"
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
curl -s -o /tmp/none.bin -D /tmp/none.hdr "http://127.0.0.1:$GW_PORT/files.dat"
grep -iE "content-encoding|content-length" /tmp/none.hdr | tr -d '\r' || true
if cmp -s /tmp/none.bin /tmp/id.bin; then
  echo "byte-identical passthrough: OK"
else
  echo "PASSTHROUGH MISMATCH"; exit 1
fi

echo "=== Range through gateway → 206 ==="
curl -s -o /tmp/rng.bin -D /tmp/rng.hdr -H "Range: bytes=100-199" "http://127.0.0.1:$GW_PORT/files.dat"
grep -iE "HTTP/|content-range|content-length" /tmp/rng.hdr | tr -d '\r'
echo "wire: $(wc -c < /tmp/rng.bin)"
# must be 206, 100 bytes, match identity slice
if ! grep -qi "206" /tmp/rng.hdr; then echo "RANGE NOT 206"; exit 1; fi
if [ "$(wc -c < /tmp/rng.bin)" != "100" ]; then echo "RANGE LEN MISMATCH"; exit 1; fi
if ! cmp -s <(dd if=/tmp/id.bin bs=1 skip=100 count=100 status=none) /tmp/rng.bin; then
  echo "RANGE BODY MISMATCH"; exit 1
fi
echo "range 206 lossless: OK"

echo SMOKE_DONE
