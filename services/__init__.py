"""services — application services layer for sloughGPT.

Not a domain. `domain/` holds substrate (cognition, core, generate, training);
`services/` holds product/integration logic that legitimately varies and depends
inward on `domain/` — never the reverse.

    auth      — Users, tenants, workspaces, RBAC
    billing   — Token grants and usage metering
    mobile    — Expo push notifications

Rules:
  * nothing under `domain/` may import `services.*` — that inverts the
    dependency direction this layer exists to enforce.
  * `services.*` may import `domain.*`.
  * `domain/api` stays in `domain/`: `domain/infrastructure/_internal/`
    consumes its SSE envelope (12 cross-package refs).
"""
