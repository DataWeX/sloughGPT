"""domain — Domain packages for sloughGPT.

Subpackages:
    memory     — Chat/task memory
    voice      — TTS, STT, phoneme encoding
    knowledge  — Fact storage, ingestion
    settings   — Persistent config
    soul       — Cognitive engine

Application services (auth, billing, mobile) live outside this package in
`services/`. Nothing under `domain/` may import `services.*` — that inverts the
dependency direction.
"""
