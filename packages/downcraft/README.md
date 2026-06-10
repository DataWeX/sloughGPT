# downcraft

Generic HTTP/HTTPS downloader with **cross-session resume** via HTTP `Range` headers. Survives power loss, process crashes, and days-long gaps between sessions.

## Quick start

```bash
pip install downcraft

# Download any URL
downcraft url https://example.com/bigfile.iso /tmp/bigfile.iso

# Or a HuggingFace model
downcraft hf Qwen/Qwen2.5-0.5B-Instruct

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

# HuggingFace model (requires huggingface-hub)
from downcraft import download_hf_model

result = download_hf_model("gpt2")
print(result["cache_dir"])  # path to HF cache
```

## CLI

| Command | Description |
|---------|-------------|
| `downcraft url <url> <dest>` | Download any file |
| `downcraft hf <model_id>` | Download HF model |
| `downcraft status <key>` | Check download status |
| `downcraft list` | List all tracked |
| `downcraft verify <key>` | Verify integrity |

## Dependencies

- **requests** (required) — HTTP with Range header support
- **huggingface-hub** (optional) — only needed for HF model commands

Install with HF support: `pip install downcraft[hf]`
