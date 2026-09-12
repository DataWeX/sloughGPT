# downcraft

Generic HTTP/HTTPS downloader with **cross-session resume** via HTTP `Range` headers. Survives power loss, process crashes, and days-long gaps between sessions.

Optional **LZ4 compression** support for bandwidth savings.

## Quick start

```bash
pip install downcraft

# Download any URL
downcraft url https://example.com/bigfile.iso /tmp/bigfile.iso

# Download with compression (server sends LZ4)
downcraft url https://example.com/bigfile.iso.lz4 /tmp/bigfile.iso --compressed

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

## Compression

Install with compression support:

```bash
pip install downcraft[compress]
```

### Compress a file

```bash
downcraft compress input.bin output.lz4
downcraft compress input.bin output.lz4 --level 9  # higher compression
```

### Decompress a file

```bash
downcraft decompress input.lz4 output.bin
```

### Check if a file is compressed

```bash
downcraft peek file.lz4
```

### Server-side: serve files with compression

```python
from downcraft import CompressedFileServer

server = CompressedFileServer()
response_data = server.serve("/path/to/model.bin")
# Use response_data["iterator"] with StreamingResponse
```

### Client-side: download with decompression

```python
from downcraft import download

# Download compressed file (auto-decompresses)
result = download(
    "https://example.com/model.bin.lz4",
    "/tmp/model.bin",
    compressed=True,
)
```

### Auto-detect compression

```python
from downcraft import auto_decompress, is_compressed_file

# Auto-detect and decompress
auto_decompress("file.lz4", "file.bin")

# Check if a file is compressed
if is_compressed_file("file.lz4"):
    print("File is LZ4 compressed")
```

### Direct compression/decompression

```python
from downcraft import (
    compress_bytes, decompress_bytes,
    compress_file, decompress_file,
    compress_stream, decompress_stream,
)

# Bytes
compressed, result = compress_bytes(b"hello world")
decompressed, _ = decompress_bytes(compressed)

# Files
compress_file("input.bin", "output.lz4")
decompress_file("output.lz4", "restored.bin")

# Streams
import io
src = io.BytesIO(b"data")
dst = io.BytesIO()
compress_stream(src, dst)
```

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
| `downcraft url <url> <dest> --compressed` | Download with LZ4 decompression |
| `downcraft status <key>` | Check download status |
| `downcraft list` | List all tracked |
| `downcraft compress <src> <dest>` | Compress file with LZ4 |
| `downcraft decompress <src> <dest>` | Decompress LZ4 file |
| `downcraft peek <file>` | Check if file is compressed |

## Dependencies

- **requests** (required) — HTTP with Range header support
- **lz4** (optional) — Compression support

Install: `pip install downcraft` or `pip install downcraft[compress]`
