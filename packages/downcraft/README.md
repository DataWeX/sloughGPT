# downcraft

Generic HTTP/HTTPS downloader with **cross-session resume** via HTTP `Range` headers. Survives power loss, process crashes, and days-long gaps between sessions.

This package is deliberately HuggingFace-agnostic — it downloads any URL. HuggingFace-specific model workflows (Hub metadata, cache layout, model download/resume/verify) live in the application layer (`domains.infrastructure.hf_hub`), composed from the generic primitives here.

## Quick start

```bash
pip install downcraft

# Download any URL
downcraft url https://example.com/bigfile.iso /tmp/bigfile.iso

# Check status
downcraft status https://example.com/bigfile.iso

# List all tracked downloads
downcraft list
```

## How resume works

1. Download starts → writes to `file.sgpart`
2. Process crashes or connection drops → `.sgpart` + `~/.downcraft/state.json` survive
3. Same URL requested again → detects `.sgpart`, sends `Range: bytes=N-` to server
4. Server sends only remaining bytes → appended to `.sgpart`
5. On completion → atomically renames to final filename

## Python API

```python
from downcraft import download

# Generic URL
result = download("https://example.com/bigfile.iso", "/tmp/bigfile.iso")
print(result["status"])  # "complete"

# With progress callback
def on_progress(downloaded, total, speed_bps):
    pct = int(downloaded / total * 100) if total else 0
    print(f"\r{pct}% @ {speed_bps/1e6:.1f} MB/s", end="")

download("https://...", "/tmp/file", on_progress=on_progress)

# Lower-level primitives
from downcraft.downloader import download_file, DownloadError
from downcraft.state import get_state
from downcraft.verify import verify_file
```

## CLI

| Command | Description |
|---------|-------------|
| `downcraft url <url> <dest>` | Download any file |
| `downcraft status <key>` | Check download status |
| `downcraft list` | List all tracked |

## Dependencies

- **requests** (required) — HTTP with Range header support

Install: `pip install downcraft`
