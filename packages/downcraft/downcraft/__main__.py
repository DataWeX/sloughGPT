"""
CLI entry point for downcraft — generic HTTP downloader with
cross-session resume.  Supports any URL that honors HTTP Range headers.

Usage::

    # Download any file by URL
    python -m downcraft url https://example.com/bigfile.iso /tmp/bigfile.iso

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

    print(f"Downloading {url}")
    print(f"  → {dest}")

    t0 = time.time()
    try:
        result = download(
            url=url,
            dest=dest,
            label=dest.rsplit("/", 1)[-1] if "/" in dest else dest,
            on_progress=lambda b, t, s: _progress("", b, t, s),
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
    print(f"Key:      {ms.model_id}")
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
        key = ms.model_id[:50]
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

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
