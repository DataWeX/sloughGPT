# Voice/TTS lane green baseline

- generated: 2026-09-23T05:11:56Z
- lane: worker-clean facade-seam consolidation

| lane              | committed         | disk==HEAD | status                           |
| ----------------- | ----------------- | ---------- | -------------------------------- |
| test_voice_router | 43f05290          | MATCH      | green (5 passed, facade-shaped)  |
| test_tts          | 43f05290          | MATCH      | green (50 passed, facade-shaped) |
| rag-facade-dedup  | 43f05290 (merged) | -          | merged into mainline             |

Stale-seam markers (prod 22050/voice-engine/browser-fallback/server_tts:True facade contract): 0 across both committed suites.

Worker overlap: apps/web (Vite) only — no domain/ seam collision.
