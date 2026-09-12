"""
CLI entry point for downcraft — generic HTTP downloader with
cross-session resume and optional LZ4 compression.  Supports any URL
that honors HTTP Range headers.

Usage::

    # Download any file by URL
    python -m downcraft url https://example.com/bigfile.iso /tmp/bigfile.iso

    # Download with LZ4 decompression
    python -m downcraft url https://example.com/bigfile.iso.lz4 /tmp/bigfile.iso --compressed

    # Estimate download size before downloading
    python -m downcraft estimate https://example.com/bigfile.iso

    # Compress a file
    python -m downcraft compress input.bin output.lz4

    # Decompress a file
    python -m downcraft decompress input.lz4 output.bin

    # Check if a file is compressed
    python -m downcraft peek file.lz4

    # Verify file integrity
    python -m downcraft verify file.bin <sha256>

    # Check status
    python -m downcraft status <url>

    # List all tracked downloads
    python -m downcraft list
"""

import argparse
import logging
import sys
import time

from . import download
from .download import state
from .resolve import resolve_page

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("downcraft")


# ---------------------------------------------------------------------------
# Shared progress renderer
# ---------------------------------------------------------------------------

def _render_progress_bar(pct: int, width: int = 30) -> str:
    filled = int(pct / 100 * width)
    bar = "█" * filled + "░" * (width - filled)
    return bar


def _progress(label: str, downloaded: int, total: int, speed: float):
    if total == 0:
        pct = 0
    else:
        pct = int(downloaded / total * 100)
    mb_dl = downloaded / (1024 * 1024)
    mb_total = total / (1024 * 1024)
    speed_mb = speed / (1024 * 1024)
    bar = _render_progress_bar(pct)
    print(
        f"\r  {bar} {mb_dl:.0f}/{mb_total:.0f} MB ({pct}%) @ {speed_mb:.1f} MB/s",
        end="",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Generic URL download
# ---------------------------------------------------------------------------

def cmd_url(args: argparse.Namespace):
    """Download any URL with resume."""
    url = args.url
    dest = args.dest
    compressed = args.compressed

    print(f"Downloading {url}")
    print(f"  → {dest}")
    if compressed:
        print(f"  → Decompressing LZ4")

    t0 = time.time()
    try:
        result = download(
            url=url,
            dest=dest,
            label=dest.rsplit("/", 1)[-1] if "/" in dest else dest,
            on_progress=lambda b, t, s: _progress("", b, t, s),
            compressed=compressed,
        )
        elapsed = time.time() - t0
        mb = (result.get("total_bytes", 0) or 0) / (1024 * 1024)
        print(f"\n✓ Done — {mb:.0f} MB in {elapsed:.0f}s")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Status / List
# ---------------------------------------------------------------------------

def cmd_status(args: argparse.Namespace):
    """Show download status for a URL."""
    key = args.key
    st = state.get_state()
    ms = st.get(key)
    if ms is None:
        print(f"{key}: not found in state")
        return

    mb_dl = ms.bytes_downloaded / (1024 * 1024)
    mb_total = ms.total_bytes / (1024 * 1024)
    print(f"Key:      {ms.key}")
    print(f"Status:   {ms.status}")
    print(f"Progress: {mb_dl:.0f} / {mb_total:.0f} MB ({ms.percentage}%)")
    print(f"Files:    {ms.files_completed}/{ms.files_total}")
    if ms.error:
        print(f"Error:    {ms.error}")


def cmd_list(args: argparse.Namespace):
    """List all tracked downloads."""
    st = state.get_state()
    models = st.list()
    if not models:
        print("No downloads tracked in state.")
        return

    for ms in models:
        mb_dl = ms.bytes_downloaded / (1024 * 1024)
        mb_total = ms.total_bytes / (1024 * 1024)
        key = ms.key[:50]
        print(f"{key:50s} {ms.status:12s} {mb_dl:8.0f}/{mb_total:.0f} MB ({ms.percentage:5.1f}%)")


# ---------------------------------------------------------------------------
# Resolve — extract real download URL from a page
# ---------------------------------------------------------------------------

def cmd_resolve(args: argparse.Namespace):
    """Resolve a page and show ranked download links."""
    url = args.url
    limit = args.limit

    print(f"Resolving {url}...")
    links = resolve_page(url, on_progress=lambda msg: print(f"  {msg}"))

    if not links:
        print("No download links found.")
        sys.exit(1)

    print(f"\nFound {len(links)} candidate(s):\n")
    for i, link in enumerate(links[:limit], 1):
        marker = " ★" if i == 1 else ""
        ext = f" ({link.extension})" if link.extension else ""
        title = f" — {link.title}" if link.title else ""
        print(f"  {i}. [{link.confidence:.2f}] {link.url}{ext}{title}{marker}")

    if args.best:
        print(f"\nBest: {links[0].url}")


# ---------------------------------------------------------------------------
# Capture — local server for browser extension
# ---------------------------------------------------------------------------

def cmd_capture(args: argparse.Namespace):
    """Start capture server for browser extension."""
    from .server import start_capture_server

    port = args.port

    print(f"Starting capture server on http://127.0.0.1:{port}")
    print(f"Extension: load extension/ folder in chrome://extensions")
    print("Press Ctrl+C to stop.\n")

    def on_capture(entry):
        print(f"  → {entry.url}")
        if entry.title:
            print(f"    {entry.title}")

    server = start_capture_server(port=port, on_capture=on_capture)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


# ---------------------------------------------------------------------------
# Compress / Decompress
# ---------------------------------------------------------------------------

def cmd_compress(args: argparse.Namespace):
    """Compress a file using LZ4."""
    from .download.compress import compress_file

    src = args.source
    dest = args.dest
    level = args.level

    print(f"Compressing {src}")
    t0 = time.time()
    try:
        result = compress_file(src, dest, compression_level=level)
        elapsed = time.time() - t0
        src_mb = result.bytes_uncompressed / (1024 * 1024)
        dst_mb = result.bytes_compressed / (1024 * 1024)
        print(f"✓ Done — {src_mb:.1f} MB → {dst_mb:.1f} MB ({result.savings_pct:.1f}% saved) in {elapsed:.1f}s")
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)


def cmd_decompress(args: argparse.Namespace):
    """Decompress an LZ4 file."""
    from .download.compress import decompress_file

    src = args.source
    dest = args.dest

    print(f"Decompressing {src}")
    t0 = time.time()
    try:
        result = decompress_file(src, dest)
        elapsed = time.time() - t0
        mb = result.bytes_uncompressed / (1024 * 1024)
        print(f"✓ Done — {mb:.1f} MB in {elapsed:.1f}s")
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)


def cmd_peek(args: argparse.Namespace):
    """Check if a file is LZ4 compressed and show header info."""
    from .download.compress import is_compressed_file, peek_compressed_header

    fpath = args.file

    if not is_compressed_file(fpath):
        print(f"{fpath}: not compressed")
        sys.exit(0)

    with open(fpath, "rb") as f:
        info = peek_compressed_header(f)

    if info:
        print(f"File:     {fpath}")
        print(f"Format:   SLZ4 (LZ4)")
        print(f"Size:     {info['uncompressed_size'] / (1024 * 1024):.1f} MB uncompressed")
        print(f"SHA-256:  {info['sha256']}")
    else:
        print(f"{fpath}: unable to read header")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Estimate — check download size before downloading
# ---------------------------------------------------------------------------

def cmd_estimate(args: argparse.Namespace):
    """Estimate download size and compression savings."""
    from .download.http import estimate_download_size

    url = args.url

    print(f"Estimating {url}...")
    info = estimate_download_size(url)

    if "error" in info:
        print(f"✗ Error: {info['error']}")
        sys.exit(1)

    total_mb = info["total_bytes"] / (1024 * 1024) if info["total_bytes"] > 0 else 0
    savings_mb = info["estimated_savings"] / (1024 * 1024) if info["estimated_savings"] > 0 else 0

    print(f"\nFile:        {url}")
    print(f"Size:        {total_mb:.1f} MB")
    print(f"Type:        {info['content_type']}")
    print(f"Encoding:    {info['content_encoding']}")
    print(f"Compressed:  {'Yes (skip double compression)' if info['is_compressed'] else 'No'}")

    if savings_mb > 0:
        print(f"Est. savings: {savings_mb:.1f} MB with LZ4 compression")
    elif info["is_compressed"]:
        print(f"Est. savings: Already compressed, no additional savings")


# ---------------------------------------------------------------------------
# Verify — check file integrity
# ---------------------------------------------------------------------------

def cmd_verify(args: argparse.Namespace):
    """Verify file integrity using SHA-256 checksum."""
    import hashlib

    fpath = args.file
    expected = args.checksum

    print(f"Verifying {fpath}")
    sha256 = hashlib.sha256()
    with open(fpath, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            sha256.update(chunk)
    actual = sha256.hexdigest()

    if expected:
        if actual == expected:
            print(f"✓ Checksum matches: {actual}")
        else:
            print(f"✗ Checksum mismatch!")
            print(f"  Expected: {expected}")
            print(f"  Actual:   {actual}")
            sys.exit(1)
    else:
        print(f"SHA-256: {actual}")


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

def main(argv: list = None):
    parser = argparse.ArgumentParser(
        prog="downcraft",
        description="Generic HTTP downloader with cross-session resume. "
                    "Supports any URL with Range headers.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # url <url> <dest>
    p_url = sub.add_parser("url", help="Download any URL")
    p_url.add_argument("url", help="HTTP/HTTPS URL")
    p_url.add_argument("dest", help="Local destination path")
    p_url.add_argument("-c", "--compressed", action="store_true", help="Expect LZ4 compression and decompress")
    p_url.set_defaults(func=cmd_url)

    # status <key>
    p_st = sub.add_parser("status", help="Check download status")
    p_st.add_argument("key", help="URL")
    p_st.set_defaults(func=cmd_status)

    # list
    p_ls = sub.add_parser("list", help="List all tracked downloads")
    p_ls.set_defaults(func=cmd_list)

    # resolve <url>
    p_res = sub.add_parser("resolve", help="Extract download links from a page")
    p_res.add_argument("url", help="Page URL to scrape")
    p_res.add_argument("-n", "--limit", type=int, default=10, help="Max results to show")
    p_res.add_argument("-b", "--best", action="store_true", help="Print only the best URL")
    p_res.set_defaults(func=cmd_resolve)

    # capture
    p_cap = sub.add_parser("capture", help="Start capture server for browser extension")
    p_cap.add_argument("-p", "--port", type=int, default=6400, help="Port (default: 6400)")
    p_cap.set_defaults(func=cmd_capture)

    # estimate <url>
    p_est = sub.add_parser("estimate", help="Estimate download size before downloading")
    p_est.add_argument("url", help="HTTP/HTTPS URL")
    p_est.set_defaults(func=cmd_estimate)

    # compress <source> <dest>
    p_compress = sub.add_parser("compress", help="Compress a file using LZ4")
    p_compress.add_argument("source", help="Source file path")
    p_compress.add_argument("dest", help="Destination file path (.lz4)")
    p_compress.add_argument("-l", "--level", type=int, default=6, help="Compression level (1-16, default: 6)")
    p_compress.set_defaults(func=cmd_compress)

    # decompress <source> <dest>
    p_decompress = sub.add_parser("decompress", help="Decompress an LZ4 file")
    p_decompress.add_argument("source", help="Compressed file path (.lz4)")
    p_decompress.add_argument("dest", help="Destination file path")
    p_decompress.set_defaults(func=cmd_decompress)

    # peek <file>
    p_peek = sub.add_parser("peek", help="Check if a file is LZ4 compressed")
    p_peek.add_argument("file", help="File to check")
    p_peek.set_defaults(func=cmd_peek)

    # verify <file> [checksum]
    p_verify = sub.add_parser("verify", help="Verify file integrity using SHA-256")
    p_verify.add_argument("file", help="File to verify")
    p_verify.add_argument("checksum", nargs="?", help="Expected SHA-256 checksum (omit to print)")
    p_verify.set_defaults(func=cmd_verify)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
