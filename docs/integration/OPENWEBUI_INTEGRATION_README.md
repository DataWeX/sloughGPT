# OpenWebUI with SLO Integration

## Overview

**Status:** Historical integration notes / design sketch. This repository does **not** currently ship the `openwebui/` tree described below. The maintained web app in this monorepo is **`apps/web/`** and the maintained API entrypoint is **`apps/api/server/main.py`**.

This document captures a potential OpenWebUI-based approach for future integration work.

## Integration Approach

**Keep OpenWebUI** - We're using the battle-tested OpenWebUI codebase as our primary web interface, adding SLO as a model provider.

**SLO Model Provider** - SLO models are integrated alongside existing OpenAI, Ollama, and other providers.

## Architecture

```
openwebui/                              # Complete OpenWebUI codebase
├── upstream/backend/open_webui/      # Core web interface
│   ├── models/
│   │   └── ollama.py           # ✅ SLO model added to provider list
│   ├── routers/
│   │   ├── models.py             # ✅ SLO appears in model list
│   ├── ollama.py             # ✅ SLO model provider added
│   └── openai.py             # Chat completions
│   └── utils/
│       └── chat.py                # ✅ Uses SLO model when selected
│
├── static/                         # CSS, JS, images
├── sripts/                       # Build and deployment scripts
├── config.py                      # Configuration management
├── alembic/                       # Database migrations
└── main.py                        # FastAPI app entry point
```

## Key Benefits

1. **Production-Ready** - OpenWebUI has been battle-tested and used in production
2. **Complete Feature Set** - Authentication, file management, user system, admin controls
3. **Multi-Provider Support** - SLO integrates seamlessly with OpenAI, Ollama, etc.
4. **RAG Capabilities** - OpenWebUI already supports vector stores and knowledge bases
5. **Mobile Responsive** - Proven mobile and tablet layouts
6. **Plugin System** - Extensible architecture for custom features
7. **Security** - Production-tested authentication and authorization

## SLO Integration Features

- **Model Registration** - SLO models appear in OpenWebUI's model selection dropdown
- **Chat Completion** - When SLO model is selected, uses our transformer directly
- **Settings Panel** - SLO-specific parameters available in settings
- **Auto-Installation** - Patches install automatically when OpenWebUI starts

## Usage (planned)

```bash
# Start OpenWebUI with SLO integration (if that external tree exists)
./openwebui/start.sh

# SLO model automatically available in:
# - Model selection dropdown
# - Chat interface
# - Settings panel  
# - RAG integration with HaulsStore
```

This gives us enterprise-grade web capabilities immediately while leveraging OpenWebUI's mature, battle-tested foundation.