"""
CLI entry point for downcraft — generic HTTP downloader with
cross-session resume.  Supports any URL plus HuggingFace model IDs.

Usage::

    # Download any file by URL
    python -m downcraft url https://example.com/bigfile.iso /tmp/bigfile.iso

    # Download a HuggingFace model (sets up all files)
    python -m downcraft hf Qwen/Qwen2.5-0.5B-Instruct

    # Check status
    python -m downcraft status <url-or-model-id>

    # List all tracked downloads
    python -m downcraft list

    # Verify integrity
    python -m downcraft verify <url-or-model-id>
"""

import argparse
import logging
import sys
import time

from . import download, hf_hub, state

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
# HuggingFace model download
# ---------------------------------------------------------------------------

def cmd_hf(args: argparse.Namespace):
    """Download a HuggingFace model."""
    model_id = args.model_id

    if hf_hub.is_download_complete(model_id, args.hf_home):
        print(f"✓ {model_id} already fully cached")
        return

    print(f"Downloading {model_id}")
    print(f"  Resolving files...", end="", flush=True)

    t0 = time.time()
    try:
        from . import download_hf_model
        result = download_hf_model(
            model_id,
            hf_home=args.hf_home,
            on_progress=lambda mid, b, t, s: _progress(mid, b, t, s),
            on_file_complete=lambda mid, fpath: print(f"\n  ✓ {fpath}"),
        )
        elapsed = time.time() - t0
        mb = (result.get("total_bytes", 0) or 0) / (1024 * 1024)
        print(f"\n✓ Downloaded {mb:.0f} MB in {elapsed:.0f}s")
        print(f"  Cache: {result.get('cache_dir', '?')}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Status / List / Verify
# ---------------------------------------------------------------------------

def cmd_status(args: argparse.Namespace):
    """Show download status for a URL or model ID."""
    key = args.key
    st = state.get_state()
    ms = st.get(key)
    if ms is None:
        # Check if it might be a cached HF model
        cached = hf_hub.is_download_complete(key, args.hf_home)
        if cached:
            print(f"{key}: fully cached (not tracked by downcraft)")
        else:
            print(f"{key}: not found in state, not cached")
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


def cmd_verify(args: argparse.Namespace):
    """Verify integrity of a downloaded file or model."""
    key = args.key
    from . import verify as vmod
    ok = vmod.verify_model(key, args.hf_home)
    if ok:
        print(f"✓ {key} integrity verified")
    else:
        print(f"✗ {key} verification FAILED")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

def main(argv: list = None):
    parser = argparse.ArgumentParser(
        prog="downcraft",
        description="Generic HTTP downloader with cross-session resume. "
                    "Supports any URL with Range headers, plus HuggingFace models.",
    )
    parser.add_argument(
        "--hf-home", default=None,
        help="HF cache directory (only for HF model commands)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # url <url> <dest>
    p_url = sub.add_parser("url", help="Download any URL")
    p_url.add_argument("url", help="HTTP/HTTPS URL")
    p_url.add_argument("dest", help="Local destination path")
    p_url.set_defaults(func=cmd_url)

    # hf <model_id>
    p_hf = sub.add_parser("hf", help="Download a HuggingFace model")
    p_hf.add_argument("model_id", help="HF model ID (e.g. gpt2)")
    p_hf.set_defaults(func=cmd_hf)

    # status <key>
    p_st = sub.add_parser("status", help="Check download status")
    p_st.add_argument("key", help="URL or model ID")
    p_st.set_defaults(func=cmd_status)

    # list
    p_ls = sub.add_parser("list", help="List all tracked downloads")
    p_ls.set_defaults(func=cmd_list)

    # verify <key>
    p_ver = sub.add_parser("verify", help="Verify integrity")
    p_ver.add_argument("key", help="URL or model ID")
    p_ver.set_defaults(func=cmd_verify)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
