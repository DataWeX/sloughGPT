# SloughGPT TUI

Standalone terminal UI for SloughGPT - OpenCode-inspired clean design.

## Installation

```bash
pip install -e ".[tui]"
# or
pip install -e .
```

Then run:
```bash
sloughgpt-tui --help
```

## Quick Start

```bash
sloughgpt-tui --local-status        # Scan models/datasets
sloughgpt-tui --api-health         # Check API
sloughgpt-tui --train --epochs 1    # Quick training
sloughgpt-tui --interactive         # Interactive menu
```

## Features

| Mode | Command | Description |
|------|---------|-------------|
| Local Status | `--local-status` | Scan repository models/datasets |
| API Health | `--api-health` | GET /health endpoint |
| API Metrics | `--api-metrics` | GET /metrics |
| Training | `--train` | Local training |
| API Training | `--train-api` | Training via API server |
| Docker | `--docker-*` | Docker compose operations |
| Interactive | `--interactive` | Menu-driven mode |

## Architecture

```
apps/tui/
├── app.py           # CLI entry point
├── components.py    # Rich UI components (OpenCode style)
├── adapters/        # Backend adapters
│   ├── http_api.py  # API client
│   ├── training.py  # Training adapters (local + HTTP)
│   ├── docker.py    # Docker compose
│   └── local_status.py
└── session.py       # Session management
```

## UI Style

OpenCode-inspired clean terminal UI:
- Minimal borders (ROUNDED)
- 256-color palette (cyan primary)
- StatusTable, ChoiceMenu, header()
- Keyboard shortcuts in menus

## Development

```bash
# Run directly
python3 -m apps.tui --local-status

# Run as installed
sloughgpt-tui --local-status
```