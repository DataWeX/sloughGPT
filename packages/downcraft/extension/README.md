# Downcraft Capture Extension

Browser extension that captures download links and sends them to your local Downcraft server.

## Install

1. Open Chrome → `chrome://extensions`
2. Enable "Developer mode"
3. Click "Load unpacked"
4. Select the `extension/` folder

## How it works

The extension intercepts clicks on file links (`.zip`, `.mp4`, `.safetensors`, etc.) and sends them to `http://localhost:6400/capture`.

The popup shows:
- Connection status (green = server online)
- Last 10 captured URLs

## Server

Start the capture server first:

```bash
downcraft capture
```

Or:

```bash
python -m downcraft capture --port 6400
```
