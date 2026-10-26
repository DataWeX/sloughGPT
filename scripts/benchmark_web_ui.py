#!/usr/bin/env python3
"""Web UI bundle baseline — measures the dist-vite build a cold visit pays for.

What it measures (card 281cf292 — regression-tracked via
`benchmark_results.py record --kind web_ui`):

  entry_js / entry_gzip       the single <script type=module> in index.html
                              (the app shell: vite-entry + static import graph)
  first_load_js / first_load_gzip
                              entry + every <link rel="modulepreload"> href —
                              bytes downloaded before first paint
  css_bytes                   stylesheets referenced by index.html
  asset_bytes / asset_count   whole dist-vite footprint (JS + CSS + rest)
  js_chunk_count              splitting granularity (routes are lazy per
                              vite/routes.ts — one chunk per route/vendor)
  largest                     top-10 JS chunks by raw bytes (name/bytes/gzip)

Gzip is computed at level 9 (same as vite's own reporter) so numbers are
comparable with `vite build` output. Invariants: routes must stay lazy (a
first_load that grows to ≈asset_bytes means the route glob went eager), and
thresholds in benchmark_results.py flag entry/first-load growth >10%.

Usage:
  ./scripts/python scripts/benchmark_web_ui.py [--build] [--out FILE]
  --build runs `npm run build --workspace apps/web` first; without it an
  existing apps/web/dist-vite is analyzed (stale dist → stale baseline).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import zlib
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent.parent / "apps" / "web"
DIST = WEB_DIR / "dist-vite"

_SCRIPT_SRC = re.compile(r'<script[^>]+type="module"[^>]+src="([^"]+)"')
_PRELOAD = re.compile(r'<link rel="modulepreload"[^>]+href="([^"]+)"')
_STYLESHEET = re.compile(r'<link rel="stylesheet"[^>]+href="([^"]+)"')


def _gzip_size(data: bytes) -> int:
    return len(zlib.compress(data, 9))


def _asset_bytes(rel: str) -> int:
    p = DIST / rel.lstrip("/")
    return p.stat().st_size if p.is_file() else 0


def _asset_gzip(rel: str) -> int:
    p = DIST / rel.lstrip("/")
    return _gzip_size(p.read_bytes()) if p.is_file() else 0


def measure() -> dict:
    index = DIST / "index.html"
    if not index.is_file():
        sys.exit(f"[ERR] {index} missing — run with --build or build apps/web first")
    html = index.read_text()

    entry = _SCRIPT_SRC.search(html)
    if not entry:
        sys.exit("[ERR] no module script in index.html")
    preloads = _PRELOAD.findall(html)
    stylesheets = _STYLESHEET.findall(html)

    assets_dir = DIST / "assets"
    js_files = sorted(assets_dir.glob("*.js"))
    css_files = sorted(assets_dir.glob("*.css"))
    all_files = [p for p in DIST.rglob("*") if p.is_file() and p.suffix != ".map"]

    first_load = [entry.group(1), *preloads]
    largest = sorted(
        (
            {
                "name": p.name,
                "bytes": p.stat().st_size,
                "gzip": _gzip_size(p.read_bytes()),
            }
            for p in js_files
        ),
        key=lambda c: c["bytes"],
        reverse=True,
    )[:10]

    return {
        "entry_js": _asset_bytes(entry.group(1)),
        "entry_gzip": _asset_gzip(entry.group(1)),
        "first_load_js": sum(_asset_bytes(r) for r in first_load),
        "first_load_gzip": sum(_asset_gzip(r) for r in first_load),
        "first_load_modules": len(first_load),
        "css_bytes": sum(_asset_bytes(r) for r in stylesheets)
        + sum(p.stat().st_size for p in css_files),
        "asset_bytes": sum(p.stat().st_size for p in all_files),
        "asset_count": len(all_files),
        "js_chunk_count": len(js_files),
        "largest": largest,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true", help="npm run build first (~90s)")
    ap.add_argument("--out", default=None, help="write metrics JSON here (default: stdout)")
    args = ap.parse_args()

    if args.build:
        print("[BUILD] npm run build --workspace apps/web …", file=sys.stderr)
        subprocess.run(
            ["npm", "run", "build", "--workspace", "apps/web"],
            cwd=WEB_DIR.parent.parent,
            check=True,
        )

    metrics = measure()
    text = json.dumps(metrics, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
        print(f"[OK] web_ui baseline → {args.out}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
